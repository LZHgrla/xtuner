# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from typing import List, Optional

import torch
from peft import PeftModel
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.model import MLPProjector
from xtuner.tools.utils import get_chat_utils
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--adapter', default=None, help='adapter name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--projector-type', default=None, help='projector type')
    parser.add_argument('--llm-weight', default=None, help='llm weight')
    parser.add_argument(
        '--projector-weight', default=None, help='projector weight')
    parser.add_argument('--image', default=None, help='image')

    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=PROMPT_TEMPLATE.default,
        help='Specify a prompt template')

    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument('--command-stop-word', default=None, help='Stop key')
    parser.add_argument('--answer-stop-word', default=None, help='Stop key')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\ndouble enter to end input (EXIT: exit chat, '
               'RESET: reset history) >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# def prepare_inputs_for_generation(
#     self,
#     input_ids,
#     past_key_values=None,
#     attention_mask=None,
#     inputs_embeds=None,
#     **kwargs
# ):
#     if past_key_values:
#         input_ids = input_ids[:, -1:]

#     position_ids = kwargs.get("position_ids", None)
#     if attention_mask is not None and position_ids is None:
#         # create position_ids on the fly for batch generation
#         position_ids = attention_mask.long().cumsum(-1) - 1
#         position_ids.masked_fill_(attention_mask == 0, 1)
#         if past_key_values:
#             position_ids = position_ids[:, -1].unsqueeze(-1)

#     # if `inputs_embeds` are passed, we only want to use them in
#     # the 1st generation step
#     if inputs_embeds is not None and past_key_values is None:
#         model_inputs = {"inputs_embeds": inputs_embeds}
#     else:
#         model_inputs = {"input_ids": input_ids}

#     model_inputs.update(
#         {
#             "position_ids": position_ids,
#             "past_key_values": past_key_values,
#             "use_cache": kwargs.get("use_cache"),
#             "attention_mask": attention_mask,
#             "images": kwargs.get("images", None)
#         }
#     )
#     return model_inputs

# def forward(self, *args, **kwargs):
#     if kwargs.get('pixel_values', None) is not None:
#         visual_outputs = self.visual_encoder(
#             kwargs['pixel_values'], output_hidden_states=True)
#         pixel_values = self.projector(
#             visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
#         kwargs['pixel_values'] = pixel_values
#         kwargs = self.prepare_inputs_labels_for_multimodal(**kwargs)


def prepare_inputs_labels_for_multimodal(
        llm,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None):
    if pixel_values is None or input_ids.shape[1] == 1:
        if (past_key_values is not None and pixel_values is not None
                and input_ids.shape[1] == 1):
            attention_mask = torch.ones(
                (attention_mask.shape[0],
                 past_key_values[-1][-1].shape[-2] + 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        return input_ids, attention_mask, past_key_values, None, labels

    new_inputs_embeds = []
    new_labels = [] if labels is not None else None
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            cur_inputs_embeds = llm.get_input_embeddings()(cur_input_ids)
            new_inputs_embeds.append(cur_inputs_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        image_token_indices = torch.where(
            cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_inputs_embeds = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_new_labels = []
            assert cur_labels.shape == cur_input_ids.shape
        while image_token_indices.numel() > 0:
            cur_pixel_values = pixel_values[cur_image_idx]
            image_token_start = image_token_indices[0]
            cur_new_inputs_embeds.append(llm.get_input_embeddings()(
                cur_input_ids[:image_token_start]))
            cur_new_inputs_embeds.append(cur_pixel_values)
            if labels is not None:
                cur_new_labels.append(cur_labels[:image_token_start])
                cur_new_labels.append(
                    torch.full((cur_pixel_values.shape[0], ),
                               IGNORE_INDEX,
                               device=labels.device,
                               dtype=labels.dtype))
                cur_labels = cur_labels[image_token_start + 1:]
            cur_image_idx += 1
            cur_input_ids = cur_input_ids[image_token_start + 1:]
            image_token_indices = torch.where(
                cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        if cur_input_ids.numel() > 0:
            cur_new_inputs_embeds.append(
                llm.get_input_embeddings()(cur_input_ids))
            if labels is not None:
                cur_new_labels.append(cur_labels)
        cur_new_inputs_embeds = [
            x.to(device=llm.device) for x in cur_new_inputs_embeds
        ]
        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds, dim=0)
        new_inputs_embeds.append(cur_new_inputs_embeds)
        if labels is not None:
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

    if any(x.shape != new_inputs_embeds[0].shape for x in new_inputs_embeds):
        max_len = max(x.shape[0] for x in new_inputs_embeds)

        new_inputs_embeds_align = []
        for cur_new_embed in new_inputs_embeds:
            cur_new_embed = torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_new_embed.shape[0],
                         cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
            new_inputs_embeds_align.append(cur_new_embed)
        new_inputs_embeds = torch.stack(new_inputs_embeds_align, dim=0)

        if labels is not None:
            new_labels_align = []
            _new_labels = new_labels
            for cur_new_label in new_labels:
                cur_new_label = torch.cat(
                    (
                        cur_new_label,
                        torch.full(
                            (max_len - cur_new_label.shape[0], ),
                            IGNORE_INDEX,
                            dtype=cur_new_label.dtype,
                            device=cur_new_label.device,
                        ),
                    ),
                    dim=0,
                )
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)

        if attention_mask is not None:
            new_attention_mask = []
            for (cur_attention_mask, cur_new_labels,
                 cur_new_labels_align) in zip(attention_mask, _new_labels,
                                              new_labels):
                new_attn_mask_pad_left = torch.full(
                    (cur_new_labels.shape[0] - labels.shape[1], ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                new_attn_mask_pad_right = torch.full(
                    (cur_new_labels_align.shape[0] -
                     cur_new_labels.shape[0], ),
                    False,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                cur_new_attention_mask = torch.cat(
                    (new_attn_mask_pad_left, cur_attention_mask,
                     new_attn_mask_pad_right),
                    dim=0)
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            assert attention_mask.shape == new_labels.shape
    else:
        new_inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)

        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full(
                (attention_mask.shape[0],
                 new_inputs_embeds.shape[1] - input_ids.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat(
                (new_attn_mask_pad_left, attention_mask), dim=1)
            assert attention_mask.shape == new_inputs_embeds.shape[:2]

    return {
        'input_ids': None,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        'inputs_embeds': new_inputs_embeds,
        'labels': new_labels
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # model_kwargs
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True
    }
    # build model
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                               **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)
    if args.adapter is not None:
        llm = PeftModel.from_pretrained(
            llm, args.adapter, offload_folder=args.offload_folder)
        print(f'Load adapter from {args.adapter}')
    visual_encoder = CLIPVisionModel.from_pretrained(args.visual_encoder)
    processor = CLIPImageProcessor.from_pretrained(args.visual_encoder)
    # TODO
    projector = MLPProjector(visual_encoder.config.hidden_size,
                             llm.config.hidden_size)
    print(f'Load projector weight from {args.projector_weight}!')
    projector_state_dict = torch.load(args.projector_weight)
    projector.load_state_dict(projector_state_dict)
    if args.llm_weight is not None:
        print(f'Load projector weight from {args.llm_weight}!')
        llm_state_dict = torch.load(args.llm_weight)
        llm.load_state_dict(llm_state_dict)
    llm.cuda()
    visual_encoder.cuda()
    projector.cuda()
    llm.eval()
    visual_encoder.eval()
    projector.eval()

    # llm.prepare_inputs_for_generation = prepare_inputs_for_generation

    if args.image is not None:
        image = Image.open(args.image).convert('RGB')
        image = expand2square(
            image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda()

    Streamer, stop_criteria = get_chat_utils(llm)
    if args.no_streamer:
        Streamer = None

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    n_turn = 0
    inputs = ''
    while True:
        text = get_input()
        while text.strip() == 'RESET':
            print('Log: History responses have been removed!')
            n_turn = 0
            inputs = ''
            text = get_input()
        if text.strip() == 'EXIT':
            print('Log: Exit!')
            exit(0)

        if args.image is not None and n_turn == 0:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        template = PROMPT_TEMPLATE[args.prompt_template]
        prompt_text = ''
        if 'SYSTEM' in template and n_turn == 0:
            system_text = None
            if args.system_template is not None:
                system_text = SYSTEM_TEMPLATE[args.system_template].format(
                    round=n_turn + 1)
            elif args.system is not None:
                system_text = args.system
            if system_text is not None:
                prompt_text += template['SYSTEM'].format(
                    system=system_text, round=n_turn + 1)
        prompt_text += template['INSTRUCTION'].format(
            input=text, round=n_turn + 1)
        inputs += prompt_text
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split('<image>')):
            if idx == 0:
                cur_encode = tokenizer(chunk)
            else:
                cur_encode = tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode['input_ids'])
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda()

        visual_outputs = visual_encoder(
            image.unsqueeze(0), output_hidden_states=True)
        pixel_values = projector(visual_outputs.hidden_states[-2][:, 1:])

        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=llm, input_ids=ids.unsqueeze(0), pixel_values=pixel_values)

        streamer = Streamer(tokenizer) if Streamer is not None else None
        generate_output = llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=streamer,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)
        if streamer is None:
            output_text = tokenizer.decode(generate_output[0][len(ids[0]):])
            end = '' if output_text[-1] == '\n' else '\n'
            print(output_text, end=end)
        inputs += tokenizer.decode(generate_output[0])
        n_turn += 1
        if len(generate_output[0]) >= args.max_new_tokens:
            print('Remove the memory of history responses, since '
                  f'it exceeds the length limitation {args.max_new_tokens}.')
            n_turn = 0
            inputs = ''


if __name__ == '__main__':
    main()
