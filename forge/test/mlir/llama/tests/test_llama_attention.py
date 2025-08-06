# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from transformers.models.llama.modeling_llama import repeat_kv
from transformers import LlamaConfig

import forge
from test.mlir.llama.utils.utils import load_attention
from forge.verify.verify import verify
from forge.config import CompilerConfig
from forge._C import DataFormat, MLIRConfig
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

    def forward(self, hidden_states, attention_mask, cos, sin, k_cache, v_cache):
        batch_size = hidden_states.size(0)
        cache = StaticCache(config=self.config, max_batch_size=batch_size, dtype=hidden_states.dtype)
        #  It is important to set dtype=hidden_states.dtype so that StaticCache dtype is always in sync with hidden_states dtype.
        #  By default, StaticCache dtype is torch.float32 so if you do update cache with torch.bfloat16 it will cast it inside the update method.

        # index 0 is because we have only one attention and one k, v cache pair
        # in general case we would have multiple pairs for each layer
        # By setting cache.key_cache[0] and cache.value_cache[0] to inputs we are bypassing the TVM constraint that
        # only tensors can be inputs to the forward. Effectively, StaticCache is now input to the forward.
        cache.key_cache[0] = k_cache
        cache.value_cache[0] = v_cache

        cache_position = torch.arange(hidden_states.size()[-2])
        position_embeddings = (cos, sin)
        # Prefill: Pass empty cache to attention forward.
        attn_output, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=cache,
            output_attentions=False,
            cache_position=cache_position,
        )
        return attn_output


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

    def forward(self, hidden_states, attention_mask, cos, sin, past_key, past_value, k_cache, v_cache):
        batch_size = hidden_states.size(0)
        past_seq_len = past_key.size(-2)

        cache = StaticCache(config=self.config, max_batch_size=batch_size, dtype=hidden_states.dtype)
        #  It is important to set dtype=hidden_states.dtype so that StaticCache dtype is always in sync with hidden_states dtype.
        #  By default, StaticCache dtype is torch.float32 so if you do update cache with torch.bfloat16 it will cast it inside the update method.

        # index 0 is because we have only one attention and one k, v cache pair
        # in general case we would have multiple pairs for each layer
        # By setting cache.key_cache[0] and cache.value_cache[0] to inputs we are bypassing the TVM constraint that
        # only tensors can be inputs to the forward. Effectively, StaticCache is now input to the forward.
        cache.key_cache[0] = k_cache
        cache.value_cache[0] = v_cache
        cache_kwargs = {"cache_position": torch.tensor(torch.arange(past_seq_len))}
        cache.update(past_key, past_value, self.layer_idx, cache_kwargs=cache_kwargs)
        # Fill static cache with past key and value tensors like it would be in prefill. Then run one decode step.
        cache_position = torch.tensor([past_seq_len])
        position_embeddings = (cos, sin)
        attn_output, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=cache,
            output_attentions=False,
            cache_position=cache_position,
        )
        return attn_output


@pytest.mark.parametrize(
    "model_path",
    [
        "openlm-research/open_llama_3b",
        "meta-llama/Llama-3.2-1B",
    ],
)
@pytest.mark.push
def test_llama_attention_prefill_mode(model_path):
    """
    Tests prefill on one attention layer of the Llama models.
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
        group = ModelGroup.GENERALITY
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
    llama_attention, config = load_attention(model_path, return_dict=True)
    # Set config for one layer attention
    config.num_hidden_layers = 1
    # Convert attention module to bfloat16
    dtype = torch.bfloat16
    llama_attention = llama_attention.to(dtype)
    layer_idx = 0
    llama_attention.layer_idx = layer_idx
    framework_model = LlamaAttentionWrapper(llama_attention, layer_idx, config)
    # framework_model = framework_model.to(dtype)

    # Set input dimensions
    batch_size = 1
    seq_len = 12
    hidden_size = config.hidden_size

    # Generate random hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    # Create lower triangular attention mask with zeros under the diagonal and -inf above the diagonal
    attention_mask = torch.triu(
        torch.full((batch_size, 1, seq_len, seq_len), -1e9, dtype=dtype),
        diagonal=1,
    )

    # Fill remaining positions with -inf
    pad_len = config.max_position_embeddings - (seq_len)
    pad_tensor = torch.full(
        (batch_size, 1, seq_len, pad_len),
        -1e9,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded_attention_mask = torch.cat([attention_mask, pad_tensor], dim=-1)

    # Generate dummy position embeddings
    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads
    num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    # Generate cos and sin for RoPE
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=dtype)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=dtype)

    # K,V cache tensors zeros:
    for i in range(config.num_hidden_layers):
        key_cache = torch.zeros(batch_size, num_key_value_heads, config.max_position_embeddings, head_dim, dtype=dtype)
        value_cache = torch.zeros(
            batch_size, num_key_value_heads, config.max_position_embeddings, head_dim, dtype=dtype
        )

    # Prepare inputs
    inputs = [hidden_states, padded_attention_mask, cos, sin, key_cache, value_cache]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "model_path",
    [
        "openlm-research/open_llama_3b",
        "meta-llama/Llama-3.2-1B",
    ],
)
@pytest.mark.push
def test_llama_attention_decode_mode(model_path):
    """
    Tests decode of one token on one attention layer of the Llama models.
    It uses StaticCache filled with past key and value tensors.
    - Tests cache update during decode on device.
    - Compares attention output for decode token but also compares key and value cache tensors after decode step.
    """

    # Extract model variant from path
    if "open_llama_3b" in model_path:
        model_name = ModelArch.OPENLLAMA
        variant = "3b"
        group = ModelGroup.GENERALITY
    elif "Llama-3.2-1B" in model_path:
        model_name = ModelArch.LLAMA3_2
        variant = "1b"
        group = ModelGroup.GENERALITY
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
    llama_attention, config = load_attention(model_path, return_dict=True)
    # Set config for one layer attention
    config.num_hidden_layers = 1
    # Convert attention module to bfloat16
    dtype = torch.bfloat16
    llama_attention = llama_attention.to(dtype)
    layer_idx = 0
    llama_attention.layer_idx = layer_idx
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
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    attention_mask = torch.zeros(batch_size, 1, seq_len, past_len + seq_len, dtype=dtype)

    # Fill remaining positions with -inf
    pad_len = config.max_position_embeddings - (past_len + seq_len)
    pad_tensor = torch.full(
        (batch_size, 1, seq_len, pad_len),
        -1e9,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded_attention_mask = torch.cat([attention_mask, pad_tensor], dim=-1)

    # Generate dummy past key/value tensors
    past_key = torch.randn(batch_size, num_key_value_heads, past_len, head_dim, dtype=dtype)
    past_value = torch.randn(batch_size, num_key_value_heads, past_len, head_dim, dtype=dtype)

    # Generate cos and sin for RoPE
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=dtype)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=dtype)

    # K,V cache tensors zeros:
    for i in range(config.num_hidden_layers):
        key_cache = torch.zeros(batch_size, num_key_value_heads, config.max_position_embeddings, head_dim, dtype=dtype)
        value_cache = torch.zeros(
            batch_size, num_key_value_heads, config.max_position_embeddings, head_dim, dtype=dtype
        )

    # Prepare inputs
    inputs = [hidden_states, padded_attention_mask, cos, sin, past_key, past_value, key_cache, value_cache]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "n_rep, cache, index, update_cache",
    [
        # Fill and repeat cache (B=1, kv_heads=2, L=8, D=16) with 2x replication
        pytest.param(
            2,
            torch.zeros(1, 2, 8, 16),
            torch.tensor([0, 1, 2, 3]),
            torch.ones(1, 2, 4, 16),
            id="fill_and_repeat",
        ),
    ],
)
@pytest.mark.push
def test_index_copy_then_repeat_kv(n_rep, cache, index, update_cache):
    class IndexCopyThenRepeatKV(torch.nn.Module):
        def __init__(self, n_rep, index):
            super().__init__()
            self.index = index
            self.n_rep = n_rep

        def forward(self, cache, update_cache):
            updated = torch.index_copy(cache, dim=2, index=self.index, source=update_cache)
            return repeat_kv(updated, self.n_rep)

    inputs = [cache, update_cache]
    model = IndexCopyThenRepeatKV(n_rep, index)
    compiled = forge.compile(model, inputs)
    verify(inputs, model, compiled)


@pytest.mark.parametrize(
    "k_cache, v_cache, k_cache_update, v_cache_update",
    [
        # Fill and repeat cache (B=1, kv_heads=2, L=8, D=16) with 2x replication
        pytest.param(
            torch.zeros(1, 2, 8, 16),
            torch.zeros(1, 2, 8, 16),
            torch.ones(1, 2, 4, 16),
            torch.ones(1, 2, 4, 16),
            id="fill_and_repeat",
        ),
    ],
)
@pytest.mark.push
def test_update_cache(k_cache, v_cache, k_cache_update, v_cache_update):
    class UpdateCache(torch.nn.Module):
        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx

        def forward(self, k_cache, v_cache, k_cache_update, v_cache_update):
            cache = StaticCache(config=self.config, max_batch_size=1, dtype=torch.float32)
            # index 0 is because we have only one k, v cache pair
            cache.key_cache[0] = k_cache
            cache.value_cache[0] = v_cache

            cache_kwargs = {"cache_position": torch.tensor(torch.arange(k_cache_update.size()[-2]))}
            cache.update(k_cache_update, v_cache_update, self.layer_idx, cache_kwargs=cache_kwargs)
            return cache.key_cache[0], cache.value_cache[0]

    model_path = "meta-llama/Llama-3.2-1B"
    config = LlamaConfig.from_pretrained(model_path)
    # Set config for one layer attention
    config.num_hidden_layers = 1
    config.max_position_embeddings = 8
    inputs = [k_cache, v_cache, k_cache_update, v_cache_update]
    model = UpdateCache(config)
    compiled = forge.compile(model, inputs)
    verify(inputs, model, compiled)
