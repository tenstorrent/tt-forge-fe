# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import DynamicCache, StaticCache

import forge
from forge.verify.verify import verify
from forge.config import CompilerConfig
from forge._C import DataFormat
from forge._C.runtime import Tensor as CTensor
from test.mlir.llama.utils.utils import load_model
from typing import List, Tuple
from loguru import logger
import time


MAX_POSITION_EMBEDDINGS = 2048
MAX_NEW_TOKENS = 20


def flatten_pastkv(past_key_values_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
    """
    Flattens a list of past key-value pairs into a single list of tensors.

    Each element in `past_key_values_list` is a tuple containing two torch.Tensors
    (typically representing a key and its corresponding value). This function
    concatenates all these tuples into one flat list, preserving their order.

    Args:
        past_key_values_list (List[Tuple[torch.Tensor, torch.Tensor]]): A list of key-value pairs,
            where each pair is represented as a tuple of two torch.Tensors.

    Returns:
        List[torch.Tensor]: A flattened list containing all the torch.Tensors from the input pairs.
    """
    new_past_key_values_list: List[torch.Tensor] = []
    for past_key_values in past_key_values_list:
        # Extend the flat list with both the key and value from the current pair.
        new_past_key_values_list.extend(list(past_key_values))
    return new_past_key_values_list


def unflatten_pastkv(past_key_values_list: List[torch.Tensor]) -> List[List[torch.Tensor]]:
    """
    Unflattens a flat list of tensors into a list of key-value pairs.

    The function assumes that the input list contains an even number of elements,
    where every two consecutive tensors form a key-value pair. For example, if the
    input list is [key1, value1, key2, value2], it returns [(key1, value1), (key2, value2)].

    Args:
        past_key_values_list (List[torch.Tensor]): A flat list of torch.Tensors.

    Returns:
        List[List[torch.Tensor]]: A list of List, each containing two torch.Tensors.

    Note:
        If the length of `past_key_values_list` is not even, assertion error will be thrown.
    """
    assert len(past_key_values_list) % 2 == 0, "past_key_values_list length should be a multiple of two"

    # Group every two consecutive tensors into a tuple.
    return [past_key_values_list[idx : idx + 2] for idx in range(0, len(past_key_values_list), 2)]


class LlamaModelWrapper(torch.nn.Module):
    """
    In LlamaModelWrapper class, forward function takes input_ids (i.e list of input token or
    or last predicted single token), attention_mask and return logits
    Args:
        1) use_cache = False
            input_id (`torch.Tensor`) - shape of (batch_size, seq_length)
            attention_mask (`torch.Tensor`) -  shape of (batch_size, seq_length)
        2) use_cache = True
            input_id (`torch.Tensor`) - shape of (batch_size, 1)
            attention_mask (`torch.Tensor`) -  shape of (batch_size, key_value_seq_len + 1)
            position_ids (`torch.Tensor`) -  shape of (batch_size, 1)
            past_key_values (`List[torch.Tensor]`)
                        key/values shape: (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)

    Returns:
        1) use_cache = False
            logits (`torch.Tensor`) - shape (batch_size, seq_length, vocab_size)
        2) use_cache = True
            logits (`torch.Tensor`) - shape (batch_size, 1, vocab_size)
            past_key_values (`List[torch.Tensor]`)
                        key/values shape: (batch_size, num_of_key_values_heads, key_value_seq_len + 1, head_dim)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_ids, attention_mask, position_ids=None, *past_key_values):

        inputs_embeds = self.embed_tokens(input_ids)

        # Formate the past key values from List(Key1, Values1, ... , KeyN, ValuesN) to
        # List(List(Key1, Values1), ... , List(KeyN, ValuesN))
        past_key_values = unflatten_pastkv(past_key_values) if len(past_key_values) > 0 else None

        if past_key_values is not None:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length()
        else:
            past_key_values_length = 0

        # The causal_mask calculated inside _update_causal_mask method in LlamaModel module
        # is not properly traced by torch.jit.trace while converting pytorch to relay ops in TVM PyTorch Frontend Conversion
        # which leads to pcc drops, because the causal mask helps to avoid attention calculation on padding token in LlamaAttention module
        # To overcome this jit trace issue, causal mask is calculated by using _prepare_4d_causal_attention_mask function.
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
        )
        outputs = self.model(
            attention_mask=causal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        model_outputs = [outputs.logits]
        if outputs.past_key_values is not None:
            # Formate the past key values from List(List(Key1, Values1), ... , List(KeyN, ValuesN)) to
            # List(Key1, Values1, ... , KeyN, ValuesN)
            model_outputs.extend(flatten_pastkv(outputs.past_key_values.to_legacy_cache()))

        return model_outputs


class LlamaModelStaticCacheWrapper(torch.nn.Module):
    """
    Wrapper for decode pass of full Llama model using StaticCache.
    Similar to LlamaDecodeStaticCacheAttentionWrapper but for the entire model.

    Forward contains:
    - initialization of StaticCache for all layers
    - Filling the cache with past key and value tensors
    - Calling model forward for one token with past key and value tensors (decode style).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.model.embed_tokens
        self.config = model.config

    def forward(self, input_ids, attention_mask, position_ids, cache_position, *args):
        """
        Args:
            input_ids: Token ids for decode step (batch_size, 1)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position ids (batch_size, 1)
            cache_position: Cache position indices (seq_len,) - avoids get_seq_length() call
            *args: k_cache, v_cache for each layer [k_cache0, v_cache0, k_cache1, v_cache1, ...]
                   Each cache tensor should be pre-populated with past key/value data
        """
        batch_size = input_ids.shape[0]
        inputs_embeds = self.embed_tokens(input_ids)

        # Parse args: every 2 tensors represent (k_cache, v_cache) for one layer
        num_layers = len(args) // 2
        assert len(args) % 2 == 0, f"Expected 2 tensors per layer, got {len(args)} tensors for {num_layers} layers"

        # Initialize StaticCache with proper config
        cache = StaticCache(config=self.config, max_batch_size=batch_size, dtype=inputs_embeds.dtype)

        # Set up cache for each layer
        for layer_idx in range(num_layers):
            base_idx = layer_idx * 2
            k_cache = args[base_idx]
            v_cache = args[base_idx + 1]

            # Set cache tensors to input tensors to bypass TVM constraint
            # This makes StaticCache effectively an input to the forward pass
            # k_cache and v_cache should already contain the past key/value data
            cache.key_cache[layer_idx] = k_cache
            cache.value_cache[layer_idx] = v_cache

        # Run model forward with StaticCache
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            inputs_embeds=inputs_embeds,
        )

        # Return only logits since StaticCache doesn't have to_legacy_cache()
        # and cache tensors will be managed externally with CTensor wrappers
        return outputs.logits


def calculate_attention_mask_and_postion_ids(
    padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
):

    # Calculate attention mask
    attention_mask = torch.zeros(padded_past_key_values_seq_length + input_seq_length, dtype=torch.long)
    attention_mask[:non_padding_past_key_values_seq_length] = 1
    attention_mask[-1] = 1
    attention_mask = attention_mask.unsqueeze(0)

    # Calculate position ids
    position_ids = torch.arange(
        non_padding_past_key_values_seq_length,
        input_seq_length + non_padding_past_key_values_seq_length,
        dtype=torch.long,
    )
    position_ids = position_ids.unsqueeze(0)

    return attention_mask, position_ids


@pytest.mark.parametrize(
    "model_path, run_on_tt_device",
    [
        ("openlm-research/open_llama_3b", False),
        ("meta-llama/Llama-3.2-1B", False),
        pytest.param(
            "openlm-research/open_llama_3b",
            True,
            marks=pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 32 GB during compile time)"
            ),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            True,
            marks=pytest.mark.xfail(
                reason="tt_metal/impl/kernels/kernel.cpp:242: tt::exception unique+common runtime args targeting kernel reader_concat_stick_layout_interleaved_start_id on (x=0,y=0) are too large. Max allowable is 256"
            ),
        ),
    ],
)
def test_decode_on_device_no_cache(model_path, run_on_tt_device):

    use_fast = False if model_path == "openlm-research/open_llama_3b" else True

    # Load Llama model and tokenizer
    model, tokenizer = load_model(model_path, use_cache=False, use_fast=use_fast)
    model.config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    framework_model = LlamaModelWrapper(model)
    framework_model.eval()

    if model_path == "openlm-research/open_llama_3b":
        tokenizer.pad_token_id = model.config.pad_token_id
    elif model_path == "meta-llama/Llama-3.2-1B":
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input sentence
    max_sequence_length = model.config.max_position_embeddings
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    non_padding_seq_len = int(torch.sum(attention_mask))
    padding_seq_len = attention_mask.shape[1] - non_padding_seq_len

    # Run Prefill on CPU with no cache to get the initial logits
    logits = framework_model(input_ids=input_ids, attention_mask=attention_mask)[0]

    # Take the last non padding token index in logits for the next token
    next_token_logits = logits[:, non_padding_seq_len - 1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    # Update the input_ids with predicted token from prefill in the first padding token index
    input_ids[:, non_padding_seq_len] = next_token
    attention_mask[:, non_padding_seq_len] = 1

    non_padding_seq_len = int(torch.sum(attention_mask))
    padding_seq_len = attention_mask.shape[1] - non_padding_seq_len

    if run_on_tt_device:
        # Compile the model on TT
        compiled_model = forge.compile(
            framework_model,
            sample_inputs=[input_ids, attention_mask],
        )

    # Run decode stage on TT device and generate tokens by appending predicted token into sequence of input tokens
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    max_new_tokens = MAX_NEW_TOKENS
    for idx in range(max_new_tokens):

        if run_on_tt_device:

            tt_inputs = [input_ids, attention_mask]

            # Run on TT device and validate TT result with Framework
            framework_output, tt_output = verify(tt_inputs, framework_model, compiled_model)

            logits = tt_output[0]

        else:

            # CPU Inference
            framework_output = framework_model(input_ids=input_ids, attention_mask=attention_mask)

            logits = framework_output[0]

        next_token_index = non_padding_seq_len + idx
        next_token_logits = logits[:, next_token_index - 1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        input_ids[:, next_token_index] = next_token
        attention_mask[:, next_token_index] = 1

    # Generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("generated_text=", generated_text)


@pytest.mark.parametrize(
    "model_path, run_on_tt_device, num_hidden_layers",
    [
        ("openlm-research/open_llama_3b", False, None),
        ("meta-llama/Llama-3.2-1B", False, None),
        pytest.param(
            "openlm-research/open_llama_3b",
            True,
            None,
            marks=[
                pytest.mark.nightly,
                pytest.mark.skip(reason="Temporarily skipping this nightly test because it takes 30GB of host memory"),
            ],
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            True,
            None,
            marks=[
                pytest.mark.push,
                # pytest.mark.skip(reason="Temporarily skipping this push test because it breaks push CI"),
            ],
        ),
        # Minimal 1-layer config for TT CI
        pytest.param(
            "openlm-research/open_llama_3b",
            True,
            1,
            marks=[
                pytest.mark.push,
                pytest.mark.skip(reason="Temporarily skipping this push test because it breaks push CI"),
            ],
        ),
    ],
)
def test_decode_on_device_cache_on_host(model_path, run_on_tt_device, num_hidden_layers):
    use_fast = False if model_path == "openlm-research/open_llama_3b" else True

    # Load model with optional override for num_hidden_layers
    model, tokenizer = load_model(
        model_path,
        use_cache=True,
        use_fast=use_fast,
        num_hidden_layers=num_hidden_layers,
    )

    model.config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    max_sequence_length = model.config.max_position_embeddings
    framework_model = LlamaModelWrapper(model)
    framework_model = framework_model.to(torch.bfloat16)
    framework_model.eval()

    if model_path == "openlm-research/open_llama_3b":
        tokenizer.pad_token_id = model.config.pad_token_id
    elif model_path == "meta-llama/Llama-3.2-1B":
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run Prefill on CPU with cache to get the initial logits and past key-values
    prefill_output = framework_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    next_token_logits = prefill_output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    generated_tokens = inputs.input_ids
    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    model_inputs = [next_token.unsqueeze(0)]
    model_inputs.extend(prefill_output[1:])

    # Zero Pad past key values in key_value_seq_len(i.e -2) dimension
    # Before padding past key values tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)
    # After Padding Past key value tensor shape -> (batch_size, num_of_key_values_heads, max_sequence_length, head_dim)
    non_padding_seq_length = prefill_output[1].shape[-2]
    for idx, past_key_or_values_states in enumerate(model_inputs[1:]):
        model_inputs[idx + 1] = torch.cat(
            [
                past_key_or_values_states,
                torch.zeros(
                    past_key_or_values_states.shape[-4],
                    past_key_or_values_states.shape[-3],
                    max_sequence_length - non_padding_seq_length,
                    past_key_or_values_states.shape[-1],
                ).to(past_key_or_values_states.dtype),
            ],
            dim=-2,
        )

    if run_on_tt_device:
        # Calculate attention mask and postion_ids
        padded_past_key_values_seq_length = model_inputs[1].shape[-2]
        input_seq_length = model_inputs[0].shape[-1]
        attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
            padded_past_key_values_seq_length, non_padding_seq_length, input_seq_length
        )

        data_format_override = DataFormat.Float16_b
        compiler_cfg = CompilerConfig(default_df_override=data_format_override)
        # Compile the model
        start = time.perf_counter()
        compiled_model = forge.compile(
            framework_model,
            sample_inputs=[model_inputs[0], attention_mask, position_ids, *model_inputs[1:]],
            compiler_cfg=compiler_cfg,
        )
        end = time.perf_counter()
        duration = end - start
        minutes = int(duration // 60)
        seconds = duration % 60

        print(f"COMPILE Block took {minutes} min {seconds:.2f} sec")

    # Run decode stage on TT device and generate tokens by passing the last predicted token and the past key values.
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    max_new_tokens = MAX_NEW_TOKENS
    for max_new_tokens_idx in range(max_new_tokens):

        non_padding_past_key_values_seq_length = non_padding_seq_length + max_new_tokens_idx
        padded_past_key_values_seq_length = model_inputs[1].shape[-2]
        input_seq_length = model_inputs[0].shape[-1]
        attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
            padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
        )

        if run_on_tt_device:
            tt_inputs = [model_inputs[0], attention_mask, position_ids, *model_inputs[1:]]

            # Run on TT device and validate TT result with Framework
            _, tt_output = verify(tt_inputs, framework_model, compiled_model)

            logits = tt_output[0]
            past_key_values = tt_output[1:]

        else:
            # CPU Inference
            framework_outputs = framework_model(model_inputs[0], attention_mask, position_ids, *model_inputs[1:])

            logits = framework_outputs[0]
            past_key_values = framework_outputs[1:]

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        model_inputs = [next_token.unsqueeze(0)]
        model_inputs.extend(past_key_values)

        # Generated new token, key and values states are appended to the past key values tensor on key_values_sequence_length dim (i.e -2).
        # For predicting next token, move the generated new token, key and values states from the last past key values tensor index
        # to the first past key values tensor padding index on key_values_sequence_length dim (i.e -2).
        for idx in range(len(model_inputs[1:])):
            model_inputs[idx + 1][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[idx + 1][:, :, -1, :]
            model_inputs[idx + 1] = model_inputs[idx + 1][:, :, :-1, :].contiguous()

        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("generated_text=", generated_text)


@pytest.mark.parametrize(
    "model_path",
    [
        "openlm-research/open_llama_3b",
        "meta-llama/Llama-3.2-1B",
    ],
)
@pytest.mark.push
def test_decode_cache_on_device(model_path):
    """
    Test Llama decode with KV cache update on device using StaticCache.

    This test:
    1. Mocks prefill on CPU to get initial past key/values
    2. Creates StaticCache wrapper for full model
    3. Tests decode step with KV cache update happening on device
    """

    if model_path == "openlm-research/open_llama_3b":
        # skip test for open_llama_3b model
        pytest.skip("Temporarily skipping, because it takes 30GB of host memory during compile")

    use_fast = False if model_path == "openlm-research/open_llama_3b" else True

    # Load model with cache enabled
    model, tokenizer = load_model(model_path, use_cache=True, use_fast=use_fast)

    # Convert to bfloat16
    dtype = torch.bfloat16
    model = model.to(dtype)
    model.eval()

    # Set up tokenizer
    if model_path == "openlm-research/open_llama_3b":
        tokenizer.pad_token_id = model.config.pad_token_id
    elif model_path == "meta-llama/Llama-3.2-1B":
        tokenizer.pad_token = tokenizer.eos_token

    # Create StaticCache wrapper
    framework_model = LlamaModelStaticCacheWrapper(model)

    # Prepare input for decode (single token)
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {inputs.input_ids}")
    print(f"Decoded prompt: {tokenizer.decode(inputs.input_ids[0])}")

    # Mock prefill on CPU to get past key/values
    prefil_model_wrapper = LlamaModelWrapper(model)
    prefill_output = prefil_model_wrapper(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    next_token_logits = prefill_output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    # Get past key/values from prefill (in legacy format)
    past_key_values_legacy = prefill_output[1:]
    past_key_values_pairs = unflatten_pastkv(past_key_values_legacy)

    # Set up decode inputs
    batch_size = 1
    seq_len = 1
    past_seq_len = inputs.input_ids.shape[1]

    # Prepare decode inputs
    decode_input_ids = next_token.unsqueeze(0)  # (1, 1)
    model.config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    max_seq_len = model.config.max_position_embeddings
    current_seq_len = past_seq_len + seq_len  # Total actual sequence length

    # Create attention mask for current sequence length
    decode_attention_mask = torch.zeros(batch_size, current_seq_len, dtype=dtype)

    # Expand to 4D: [batch_size, 1, 1, seq_len]
    decode_attention_mask = decode_attention_mask.unsqueeze(1).unsqueeze(2)

    # Pad attention mask to max_seq_len
    pad_len = max_seq_len - current_seq_len
    if pad_len > 0:
        pad_tensor = torch.full(
            (batch_size, 1, 1, pad_len),
            -1e9,
            dtype=decode_attention_mask.dtype,
            device=decode_attention_mask.device,
        )
        decode_attention_mask = torch.cat([decode_attention_mask, pad_tensor], dim=-1)

    decode_position_ids = torch.tensor([[past_seq_len]], dtype=dtype)

    num_layers = model.config.num_hidden_layers
    num_key_value_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Create initial cache_position (for the first decode step after prefill)
    decode_cache_position = torch.tensor([past_seq_len], dtype=torch.long)

    decode_inputs = [decode_input_ids, decode_attention_mask, decode_position_ids, decode_cache_position]

    # Add k_cache, v_cache for each layer (pre-populated with past key/value data)
    for layer_idx in range(num_layers):
        past_key, past_value = past_key_values_pairs[layer_idx]

        # Convert past key/values to proper dtype
        past_key = past_key.to(dtype)
        past_value = past_value.to(dtype)

        # Create cache tensors (max sequence length) and populate with past data
        k_cache = torch.zeros(batch_size, num_key_value_heads, max_seq_len, head_dim, dtype=dtype)
        v_cache = torch.zeros(batch_size, num_key_value_heads, max_seq_len, head_dim, dtype=dtype)

        # Copy past key/value data into the cache tensors at the beginning
        past_seq_len = past_key.size(-2)
        k_cache[:, :, :past_seq_len, :] = past_key
        v_cache[:, :, :past_seq_len, :] = past_value

        decode_inputs.extend([k_cache, v_cache])

    # Compile model
    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    compiled_model = forge.compile(framework_model, decode_inputs, compiler_cfg=compiler_cfg)

    # For debugging: Keep all tensors as torch tensors to test framework model directly
    # Skip CTensor conversion if testing with framework_model instead of compiled_model
    cache_start_idx = 4  # input_ids, attention_mask, position_ids, cache_position
    for i in range(cache_start_idx, len(decode_inputs)):
        decode_inputs[i] = CTensor(decode_inputs[i])

    # Initialize generated tokens with input + first predicted token
    generated_tokens = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=-1)

    # Run decode loop
    max_new_tokens = MAX_NEW_TOKENS  # Generate up to max_new_tokens more tokens
    current_seq_len = current_seq_len + 1

    start = time.perf_counter()

    for decode_step in range(max_new_tokens):
        logits = compiled_model(*decode_inputs)

        # Get next token
        logits_tensor = logits if not isinstance(logits, list) else logits[0]
        next_token_logits = logits_tensor[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        # Check for end of sequence
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append generated token
        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)
        current_seq_len += 1
        # Stop if we reach max sequence length
        if current_seq_len >= max_seq_len:
            break
        # Update decode inputs for next step
        decode_inputs[0] = next_token.unsqueeze(0)
        # update mask with 0 at new cache position
        decode_attention_mask[:, :, :, current_seq_len] = 0  # Update attention mask
        decode_inputs[1] = decode_attention_mask
        decode_inputs[2] = torch.tensor([[current_seq_len - 1]], dtype=dtype)  # Update position ids
        decode_inputs[3] = torch.tensor([current_seq_len - 1], dtype=torch.long)  # Update cache position

    end = time.perf_counter()
    duration = end - start
    minutes = int(duration // 60)
    seconds = duration % 60
    print(f"DECODE LOOP Block took {minutes} min {seconds:.2f} sec")
    # Decode and print generated text
    print(f"Time per token: {duration / (decode_step + 1):.4f} seconds")
    print(f" Num tokens generated: {decode_step + 1}")
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
