# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmengine import MessageHub

from .triton_kernels import apply_rotary_emb

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    SUPPORT_FLASH2 = True
except ImportError:
    pass


class InternLM2RotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=1000000,
                 device=None):
        super().__init__()
        self.inv_freq = 1.0 / (
            base**(torch.arange(0, dim, 2).float().to(device) / dim))

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if (seq_len > self.max_seq_len_cached
                or self.cos_cached.device != x.device
                or self.cos_cached.dtype != x.dtype):
            self.max_seq_len_cached = seq_len
            assert self.inv_freq.dtype == torch.float32
            t = torch.arange(
                self.max_seq_len_cached,
                device=x.device,
                dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(t.device))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().to(x.dtype)
            self.sin_cached = emb.sin().to(x.dtype)
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    sin = sin.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    # print(q.shape, cos.shape, rotate_half(q).shape)
    if q.size(2) == 1:
        q_embed = (q * cos[:, :, -1, :]) + (rotate_half(q) * sin[:, :, -1, :])
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    if k.size(2) == 1:
        k_embed = (k * cos[:, :, -1, :]) + (rotate_half(k) * sin[:, :, -1, :])
    else:
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def internlm2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37'
            'Please make sure use `attention_mask` instead.`')

    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # use flash attention implemented by pytorch
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=attention_mask)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def internlm2_mmca_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37'
            'Please make sure use `attention_mask` instead.`')

    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    assert len(attention_mask) == 2
    for attention_mask_one in attention_mask:
        if attention_mask_one.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError('Attention mask should be of size '
                             f'{(bsz, 1, q_len, kv_seq_len)}, '
                             f'but is {attention_mask_one.size()}')

    attention_mask_img = attention_mask[0]
    attention_mask_text = attention_mask[1]

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    attn_weights_img = attn_weights + attention_mask_img
    attn_weights_text = attn_weights + attention_mask_text

    attn_weights_img = nn.functional.softmax(
        attn_weights_img, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_text = nn.functional.softmax(
        attn_weights_text, dim=-1, dtype=torch.float32).to(query_states.dtype)
    img_valid_mask = (attention_mask_img > torch.finfo(
        attention_mask_img.dtype).min).expand(attn_weights_img.shape)
    attn_weights_img[~img_valid_mask] = 0

    attn_weights = attn_weights_img + attn_weights_text

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


# def internlm2_mmca_attn_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     **kwargs,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
#            Optional[Tuple[torch.Tensor]]]:
#     if 'padding_mask' in kwargs:
#         warnings.warn(
#             'Passing `padding_mask` is deprecated and will be removed in v4.37'
#             'Please make sure use `attention_mask` instead.`')

#     bsz, q_len, _ = hidden_states.size()

#     qkv_states = self.wqkv(hidden_states)

#     qkv_states = rearrange(
#         qkv_states,
#         'b q (h gs d) -> b q h gs d',
#         gs=2 + self.num_key_value_groups,
#         d=self.head_dim,
#     )

#     query_states = qkv_states[..., :self.num_key_value_groups, :]
#     query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
#     key_states = qkv_states[..., -2, :]
#     value_states = qkv_states[..., -1, :]

#     query_states = query_states.transpose(1, 2)
#     key_states = key_states.transpose(1, 2)
#     value_states = value_states.transpose(1, 2)

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]
#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
#                                                     cos, sin, position_ids)

#     if past_key_value is not None:
#         # reuse k, v, self_attention
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)

#     assert len(attention_mask) == 2
#     for attention_mask_one in attention_mask:
#         if attention_mask_one.size() != (bsz, 1, q_len, kv_seq_len):
#             raise ValueError('Attention mask should be of size '
#                              f'{(bsz, 1, q_len, kv_seq_len)}, '
#                              f'but is {attention_mask_one.size()}')

#     attention_mask_img = attention_mask[0]
#     attention_mask_text = attention_mask[1]

#     img_valid_mask = (attention_mask_img > torch.finfo(
#         attention_mask_img.dtype).min).any(
#             dim=-1, keepdim=True).expand(bsz, self.num_heads, q_len,
#                                          self.head_dim)

#     # use flash attention implemented by pytorch
#     if q_len == kv_seq_len:
#         indices = torch.arange(q_len)
#         attention_mask_img[:, :, indices, indices] = 0

#     attn_output_img = F.scaled_dot_product_attention(
#         query_states, key_states, value_states, attn_mask=attention_mask_img)

#     attn_output_text = F.scaled_dot_product_attention(
#         query_states, key_states, value_states, attn_mask=attention_mask_text)

#     attn_output = attn_output_text.clone()

#     attn_output[img_valid_mask] = attn_output[
#         img_valid_mask] + attn_output_img[img_valid_mask]

#     attn_output = attn_output.transpose(1, 2).contiguous()
#     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#     attn_output = self.wo(attn_output)

#     # Due to the implementation of the PyTorch version of flash attention,
#     # even when the output_attentions flag is set to True, it is not possible
#     # to return the attn_weights.
#     return attn_output, None, past_key_value


def internlm2_varlen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # Modified from https://huggingface.co/internlm/internlm-7b/blob/939a68c0dc1bd5f35b63c87d44af05ce33379061/modeling_internlm.py#L161  # noqa:E501

    is_training = self.training

    message_hub = MessageHub.get_instance('varlen_attn_args')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    indexes = message_hub.get_info(f'indexes_rank_{rank}')
    max_seqlen = message_hub.get_info(f'max_seqlen_rank_{rank}')
    assert is_training == (cumulative_len is not None)

    bsz, q_len, _ = hidden_states.size()

    assert bsz == 1, (f'If utilizing local attention, the batch size should be'
                      f' set to 1, but got {bsz}')

    qkv_states = self.wqkv(hidden_states)
    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    kv_seq_len = key_states.shape[-3]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if is_training:
        cos, sin = self.rotary_emb(value_states, max_seqlen)
        query_states = apply_rotary_emb(query_states, cos[indexes].squeeze(0),
                                        sin[indexes].squeeze(0))
        key_states = apply_rotary_emb(key_states, cos[indexes].squeeze(0),
                                      sin[indexes].squeeze(0))
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    assert SUPPORT_FLASH2
    if is_training:
        q_unpad, k_unpad, v_unpad = query_states.flatten(
            0, 1), key_states.flatten(0, 1), value_states.flatten(0, 1)
        cumulative_len = torch.cat(cumulative_len, dim=0)
        attn_output = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cumulative_len,
            cumulative_len,
            max_seqlen,
            max_seqlen,
            0,
            return_attn_probs=False,
            causal=True,
        )
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, causal=True)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def _prepare_mmca_decoder_attention_mask(self, attention_mask, input_shape,
                                         inputs_embeds,
                                         past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

    def _make_causal_mask(input_ids_shape: torch.Size,
                          dtype: torch.dtype,
                          device: torch.device,
                          past_key_values_length: int = 0):
        """Make causal mask used for bi-directional self-attention."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len),
                          torch.tensor(torch.finfo(dtype).min, device=device),
                          device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1),
                          0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(
                    tgt_len,
                    past_key_values_length,
                    dtype=dtype,
                    device=device), mask
            ],
                             dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                             tgt_len + past_key_values_length)

    def _expand_mmca_mask(mask: torch.Tensor,
                          dtype: torch.dtype,
                          tgt_len: Optional[int] = None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        # image mask
        mask_img = mask.clone()
        mask_img[mask_img != 2] = 0
        mask_img[mask_img == 2] = 1

        expanded_mask_img = mask_img[:, None,
                                     None, :].expand(bsz, 1, tgt_len,
                                                     src_len).to(dtype)

        inverted_mask_img = 1.0 - expanded_mask_img
        inverted_mask_img = inverted_mask_img.masked_fill(
            inverted_mask_img.to(torch.bool),
            torch.finfo(dtype).min)
        # text mask
        mask_text = mask.clone()
        mask_text[mask_text != 1] = 0
        mask_text[mask_text == 1] = 1
        expanded_mask_text = mask_text[:, None,
                                       None, :].expand(bsz, 1, tgt_len,
                                                       src_len).to(dtype)

        inverted_mask_text = 1.0 - expanded_mask_text
        inverted_mask_text = inverted_mask_text.masked_fill(
            inverted_mask_text.to(torch.bool),
            torch.finfo(dtype).min)

        # image tokens does not attennd to image tokens
        if tgt_len == src_len:
            # TODO: basically, the prompt phase, need to revisit this part
            for i in range(bsz):
                for j in range(tgt_len):
                    if mask[i, j] == 2:
                        inverted_mask_img[i, :, j, :] = torch.finfo(dtype).min
                        inverted_mask_text[i, :, j, :] = torch.finfo(dtype).min
                        inverted_mask_text[i, :, j, j] = 0

        return [inverted_mask_img, inverted_mask_text]  # return two masks

    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mmca_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        # if cross attention, we have two masks, this is from _expand_mask
        expanded_attn_mask = [
            expanded_attn_mask[0].to(inputs_embeds.device),
            expanded_attn_mask[1].to(inputs_embeds.device)
        ]
        if combined_attention_mask is None:
            combined_attention_mask = expanded_attn_mask
        else:
            combined_attention_mask = (expanded_attn_mask[0] +
                                       combined_attention_mask,
                                       expanded_attn_mask[1] +
                                       combined_attention_mask)

    return combined_attention_mask
