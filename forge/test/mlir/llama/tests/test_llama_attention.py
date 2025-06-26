# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_attention_layer_only
from forge.verify.verify import verify, DeprecatedVerifyConfig
from forge.forge_property_utils import Framework, ModelArch, Source, Task, ModelGroup, record_model_properties
from transformers.cache_utils import StaticCache


class LlamaAttentionWrapper(torch.nn.Module):
    """
    Wrapper for prefil pass of Llama attention module.
    Forward contains:
    - initialization of StaticCache
    - Calling attention forward for prefill sequence.
    """

    def __init__(self, attention_module, layer_idx, config):
        super().__init__()
        self.attn = attention_module
        self.layer_idx = layer_idx
        self.config = config

    def forward(self, hidden_states, attention_mask, position_ids):
        batch_size = hidden_states.size(0)
        cache = StaticCache(config=self.config, batch_size=batch_size)
        cache_position = position_ids.squeeze(0)
        # Prefill: Pass empty cache to attention forward.
        attn_output, _, updated_kv = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=False,  # disable cache to avoid returning non-Tensor
            output_attentions=False,
            cache_position=cache_position,
        )
        k_cache, v_cache = updated_kv.key_cache[self.layer_idx], updated_kv.value_cache[self.layer_idx]
        return attn_output, k_cache, v_cache


class LlamaDecodeStaticCacheAttentionWrapper(torch.nn.Module):
    """
    Wrapper for decode pass of Llama attention module.
    Forward contains:
    - initialization of StaticCache
    - Filling the cache with past key and value tensors
    - Calling attention forward for one token with past key and value tensors (decode style).
    """

    def __init__(self, attention_module, layer_idx, config):
        super().__init__()
        self.attn = attention_module
        self.layer_idx = layer_idx
        self.config = config

    def forward(self, hidden_states, attention_mask, position_ids, past_key, past_value):
        batch_size = hidden_states.size(0)
        past_seq_len = past_key.size(-2)
        max_seq_len = past_seq_len + hidden_states.size(1)

        # Initialize StaticCache with pre-allocated memory
        cache = StaticCache(config=self.config, batch_size=batch_size)
        cache_kwargs = {"cache_position": torch.tensor(torch.arange(past_seq_len))}
        cache.update(past_key, past_value, self.layer_idx, cache_kwargs=cache_kwargs)
        cache_position = torch.tensor([past_seq_len])
        attn_output, _, updated_kv = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=False,  # disable cache to avoid returning non-Tensor
            output_attentions=False,
            cache_position=cache_position,
        )
        k_cache, v_cache = updated_kv.key_cache[self.layer_idx], updated_kv.value_cache[self.layer_idx]
        return attn_output, k_cache, v_cache


@pytest.mark.parametrize(
    "model_path",
    [
        "openlm-research/open_llama_3b",
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            marks=pytest.mark.xfail(
                reason="RuntimeError: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 8623072 B which is beyond max L1 size of 1499136 B"
            ),
        ),
    ],
)
@pytest.mark.push
def test_llama_attention_prefill_mode(model_path):
    """
    This one tests prefill on one attention layer of the Llama models.
    It uses StaticCache to store key and value tensors.
    - Tests cache filling during prefill on device.
    - Compares attention output but also compares key and value cache tensors after prefill.
    """

    # Extract model variant from path
    if "open_llama_3b" in model_path:
        model_name = ModelArch.OPENLLAMA
        variant = "3b"
        group = ModelGroup.GENERALITY
    elif "Llama-3.2-1B" in model_path:
        model_name = ModelArch.LLAMA3_2
        variant = "1b"
        group = ModelGroup.RED
    else:
        model_name = ModelArch.LLAMA
        variant = "unknown"
        group = ModelGroup.GENERALITY

    # Record model details
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=model_name,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.TEXT_GENERATION,
        group=group,
    )

    # Load Llama model and tokenizer
    layer_idx = 0
    llama_attention, config = load_attention_layer_only(model_path, layer_idx=layer_idx)
    framework_model = LlamaAttentionWrapper(llama_attention, layer_idx, config)

    # Set input dimensions
    batch_size = 1
    seq_len = 12
    hidden_size = config.hidden_size

    # Generate random hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    # Create lower triangular attention mask with zeros under the diagonal and -inf above the diagonal
    attention_mask = torch.triu(torch.full((batch_size, 1, seq_len, seq_len), float("-inf")), diagonal=1)

    # Fill remaining positions with -inf
    pad_len = config.max_position_embeddings - (seq_len)
    pad_tensor = torch.full(
        (batch_size, 1, seq_len, pad_len),
        float("-inf"),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded_attention_mask = torch.cat([attention_mask, pad_tensor], dim=-1)

    # Optional position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Prepare inputs
    inputs = [hidden_states, padded_attention_mask, position_ids]

    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "model_path",
    [
        "openlm-research/open_llama_3b",
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            marks=pytest.mark.xfail(
                reason="RuntimeError: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 8623072 B which is beyond max L1 size of 1499136 B"
            ),
        ),
    ],
)
@pytest.mark.push
def test_llama_attention_decode_mode(model_path):
    """
    This one tests decode on one attention layer of the Llama models.
    It uses StaticCache filled with past key and value tensors.
    - Tests cache update during decode on device.
    - Compares attention outpu for decode token but also compares key and value cache tensors after decode step.
    """

    # Extract model variant from path
    if "open_llama_3b" in model_path:
        model_name = ModelArch.OPENLLAMA
        variant = "3b"
        group = ModelGroup.GENERALITY
    elif "Llama-3.2-1B" in model_path:
        model_name = ModelArch.LLAMA3_2
        variant = "1b"
        group = ModelGroup.RED
    else:
        model_name = ModelArch.LLAMA
        variant = "unknown"
        group = ModelGroup.GENERALITY

    # Record model details
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=model_name,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.TEXT_GENERATION,
        group=group,
    )

    # Load Llama model and tokenizer
    llama_attention, config = load_attention_layer_only(model_path)
    layer_idx = 0
    framework_model = LlamaDecodeStaticCacheAttentionWrapper(llama_attention, layer_idx, config)

    # Set input dimensions
    batch_size = 1
    seq_len = 1
    past_len = 12
    hidden_size = config.hidden_size
    num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads

    # Generate random hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.zeros(batch_size, 1, seq_len, past_len + seq_len)

    # Fill remaining positions with -inf
    pad_len = config.max_position_embeddings - (past_len + seq_len)
    pad_tensor = torch.full(
        (batch_size, 1, seq_len, pad_len),
        float("-inf"),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded_attention_mask = torch.cat([attention_mask, pad_tensor], dim=-1)

    # Generate dummy past key/value tensors
    past_key = torch.randn(batch_size, num_key_value_heads, past_len, head_dim)
    past_value = torch.randn(batch_size, num_key_value_heads, past_len, head_dim)

    # Optional position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Prepare inputs
    inputs = [hidden_states, padded_attention_mask, position_ids, past_key, past_value]


    verify_cfg = DeprecatedVerifyConfig()
    verify_cfg.verify_all = True
    verify_cfg.enable_op_level_comparision = True
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name, verify_cfg=verify_cfg)

    verify(inputs, framework_model, compiled_model)
