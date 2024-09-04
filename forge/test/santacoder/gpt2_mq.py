# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""PyTorch OpenAI GPT-2 model modified with MultiQuery attention"""


import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from modeling_gpt2 import GPT2Model, GPT2Block, GPT2PreTrainedModel, GPT2LMHeadModel, GPT2SequentialCaller
from configuration_gpt2_mq import GPT2CustomConfig, MULTI_QUERY, MULTI_HEAD



class GPT2MQAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        assert config.attention_head_type == MULTI_QUERY

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        if is_cross_attention:
            raise NotImplementedError("Cross-attention not implemented for MQA")
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            # self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
            # Keys and values are shared across heads
            self.kv_attn = Conv1D(2 * self.head_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # query: (b, num_heads * sq, head_dim)
        # key: (b, head_dim, sk)
        # value: (b, sk, head_dim)
        batch_size = query.size(0)
        query_length = query.size(1) // self.num_heads
        key_length = key.size(2)
        # (b, num_heads * sq, head_dim) x (b, head_dim, sk) -> (b, num_heads * sq, sk)
        attn_weights = torch.matmul(query, key)
        # -> (b, num_heads, sq, sk)
        attn_weights = attn_weights.view(batch_size, self.num_heads, query_length, key_length)

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # (b, num_heads, sq, sk) -> (b, num_heads * sq, sk)
        _attn_weights = attn_weights.view(batch_size, self.num_heads, query_length, key_length)
        # (b, num_heads * sq, sk) x (b, sk, head_dim) -> (b, num_heads * sq, head_dim)
        attn_output = torch.matmul(_attn_weights, value)
        attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)

        return attn_output, attn_weights

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        attention_mask: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention not implemented for MQA")
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query = self.q_attn(hidden_states)
            key, value = self.kv_attn(hidden_states).split(self.head_dim, dim=2)


        batch_size, seq_length = query.shape[:2]
        # (query_length, batch, num_heads, head_dim)
        # (batch, num_heads * query_length, head_dim)\

        # (batch, query_length, hidden_size) -> (batch, num_heads, query_length, head_dim)
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        # -> (batch, num_heads * query_length, head_dim)
        #query = query.reshape(batch_size, self.num_heads * seq_length, self.head_dim)

        # (batch, query_length, hidden_size) -> (batch, query_length * num_heads, head_dim)
        # query = query.view(
        #     batch_size, seq_length, self.num_heads, self.head_dim,
        # ).reshape(
        #     batch_size, seq_length * self.num_heads, self.head_dim
        # )
        key = key.permute(0, 2, 1)  # (batch_size, head_dim, seq_length)
        # value (batch_size, seq_length, head_dim)

        assert past_value is not None, "past_key and past_value should be passed together"
        # Fill in on sequence dimension

        assert key.shape[0] == 1, "single batch only for decode right now"

        # Position doesn't matter so we'll add it to the start. This should be attended to in the mask.
        full_key = torch.cat((past_key, key), dim=-1)
        full_value = torch.cat((past_value, value), dim=-2)
        
        if self.reorder_and_upcast_attn:
            raise NotImplementedError("Reorder and upcast attention not implemented for MQA")
        else:
            attn_output, attn_weights = self._attn(query, full_key, full_value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, key, value) # just returning the new key and value
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, key, value, (attentions)


# inherit from gpt_modeling.py, and override `attn` module
class GPT2CustomBlock(GPT2Block):

    def __init__(self, config: GPT2CustomConfig, layer_idx=None):
        super().__init__(config, layer_idx)
        # Override attention module if using multiquery
        if config.attention_head_type == MULTI_QUERY:
            self.attn = GPT2MQAttention(config, layer_idx=layer_idx)
            if config.add_cross_attention:
                raise NotImplementedError("Cross-attention not implemented for MQA")


# inherit from gpt_modeling.py and override `__init__` method
class GPT2CustomModel(GPT2Model):
    config_class = GPT2CustomConfig
    
    def __init__(self, config):
        GPT2PreTrainedModel.__init__(self, config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2CustomBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.blocks = GPT2SequentialCaller(self.h)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GPT2LMHeadCustomModel(GPT2LMHeadModel):
    config_class = GPT2CustomConfig

    def __init__(self, config):
        GPT2PreTrainedModel.__init__(self, config)
        self.transformer = GPT2CustomModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
