# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .modules import dispatch_modules
from .utils import LoadWoInit, find_all_linear_names, traverse_dict


class LinearProjector(nn.Module):

    def __init__(self, visual_hidden_size, llm_hidden_size):
        super().__init__()
        self.layers = nn.Linear(visual_hidden_size, llm_hidden_size)
        self.use_activation_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.use_activation_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.use_activation_checkpointing = False

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.layers.register_forward_hook(make_inputs_require_grad)

    def forward(self, x):
        if self.use_activation_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.layers, x)
        else:
            layer_outputs = self.layers(x)
        return layer_outputs


class MLPProjector(nn.Module):

    def __init__(self, visual_hidden_size, llm_hidden_size, depth=2):
        super().__init__()
        modules = [nn.Linear(visual_hidden_size, llm_hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
        self.layers = nn.Sequential(*modules)
        self.use_activation_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.use_activation_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.use_activation_checkpointing = False

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.layers.register_forward_hook(make_inputs_require_grad)

    def forward(self, x):
        if self.use_activation_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.layers, x)
        else:
            layer_outputs = self.layers(x)
        return layer_outputs


class LLaVAModel(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm,
                 freeze_visual_encoder,
                 visual_select_layer=-2,
                 projector_path=None,
                 llm_lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True):
        super().__init__()
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)
        self.projector = MLPProjector(self.visual_encoder.config.hidden_size,
                                      self.llm.config.hidden_size)
        if projector_path is not None:
            projector_state_dict = torch.load(projector_path)
            self.load_state_dict(projector_state_dict, strict=False)
            print(f'Load {projector_path}')

        if freeze_llm:
            self.llm.requires_grad_(False)
        if freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)

            # enable llm gradient checkpointing for memory efficiency
            self.llm.gradient_checkpointing_enable()

            # For backward compatibility
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)

            # enable llm gradient checkpointing for memory efficiency
            self.visual_encoder.gradient_checkpointing_enable()

            # For backward compatibility
            self.projector.enable_input_require_grads()
            # enable projector gradient checkpointing for memory efficiency
            self.projector.gradient_checkpointing_enable()

        if isinstance(llm_lora, dict) or isinstance(
                llm_lora, Config) or isinstance(llm_lora, ConfigDict):
            self.llm_lora = BUILDER.build(llm_lora)
        else:
            self.llm_lora = llm_lora
        self.peft_model = peft_model
        self.use_lora = llm_lora is not None
        if self.use_lora:
            self._prepare_for_lora(peft_model, use_activation_checkpointing)

        self._is_init = True

        self.visual_select_layer = visual_select_layer

    def _prepare_for_lora(self,
                          peft_model=None,
                          use_activation_checkpointing=True):
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if self.llm_lora.target_modules is None:
            modules = find_all_linear_names(self.llm)
            self.llm_lora.target_modules = modules

        self.llm = get_peft_model(self.llm, self.llm_lora)
        if peft_model is not None:
            _ = load_checkpoint(self, peft_model)

    def init_weights(self):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict()
        to_return = {}
        for k, v in state_dict.items():
            if 'projector' in k:
                to_return[k] = v
        return OrderedDict(to_return)

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):
        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'], output_hidden_states=True)
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            data['pixel_values'] = pixel_values
            data = self.prepare_inputs_labels_for_multimodal(**data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    # def prepare_inputs_for_generation(self, **kwargs):
    #     model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
    #     model_inputs.update({})
    #     return model_inputs

    def prepare_inputs_labels_for_multimodal(
            self,
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
                cur_inputs_embeds = self.llm.get_input_embeddings()(
                    cur_input_ids)
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
                cur_new_inputs_embeds.append(self.llm.get_input_embeddings()(
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
                    self.llm.get_input_embeddings()(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_inputs_embeds = [
                x.to(device=self.device) for x in cur_new_inputs_embeds
            ]
            cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds, dim=0)
            new_inputs_embeds.append(cur_new_inputs_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_inputs_embeds[0].shape
               for x in new_inputs_embeds):
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
