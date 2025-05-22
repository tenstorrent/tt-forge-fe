# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import forge
from forge.verify.verify import verify
from test.mlir.llama.utils.utils import load_model
from typing import List, Tuple
from loguru import logger


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

        # The causal_mask calculated inside _update_causal_mask method in LlamaModel module
        # is not properly traced by torch.jit.trace while converting pytorch to relay ops in TVM PyTorch Frontend Conversion
        # which leads to pcc drops, because the causal mask helps to avoid attention calculation on padding token in LlamaAttention module
        # To overcome this jit trace issue, causal mask is calculated by using _prepare_4d_causal_attention_mask function.
        past_key_values_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
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
            model_outputs.extend(flatten_pastkv(outputs.past_key_values))

        return model_outputs


def calculate_attention_mask_and_postion_ids(
    padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
):

    # Calculate attention mask
    attention_mask = torch.zeros(padded_past_key_values_seq_length + input_seq_length)
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
def test_llama_prefill_on_cpu_decode_on_tt_no_cache(model_path, run_on_tt_device):

    use_fast = False if model_path == "openlm-research/open_llama_3b" else True

    # Load Llama model and tokenizer
    model, tokenizer = load_model(model_path, return_dict=True, use_cache=False, use_fast=use_fast)
    framework_model = LlamaModelWrapper(model)
    framework_model.eval()

    if model_path == "openlm-research/open_llama_3b":
        tokenizer.pad_token_id = model.config.pad_token_id
    elif model_path == "meta-llama/Llama-3.2-1B":
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input sentence
    max_sequence_length = 58
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_sequence_length,
        pad_to_max_length=True,
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
    max_new_tokens = padding_seq_len
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
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
    ],
)
def test_llama_prefill_on_cpu_decode_on_tt_cache(model_path, run_on_tt_device):

    use_fast = False if model_path == "openlm-research/open_llama_3b" else True

    # Load Llama model and tokenizer
    model, tokenizer = load_model(model_path, return_dict=True, use_cache=True, use_fast=False)
    framework_model = LlamaModelWrapper(model)
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
    # After Padding Past key value tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len + max_new_tokens, head_dim)
    max_new_tokens = 46
    non_padding_seq_length = prefill_output[1].shape[-2]
    for idx, past_key_or_values_states in enumerate(model_inputs[1:]):
        model_inputs[idx + 1] = torch.cat(
            [
                past_key_or_values_states,
                torch.zeros(
                    past_key_or_values_states.shape[-4],
                    past_key_or_values_states.shape[-3],
                    max_new_tokens,
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

        # Compile the model
        compiled_model = forge.compile(
            framework_model,
            sample_inputs=[model_inputs[0], attention_mask, position_ids, *model_inputs[1:]],
        )

    # Run decode stage on TT device and generate tokens by passing the last predicted token and the past key values.
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
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
            framework_output, tt_output = verify(tt_inputs, framework_model, compiled_model)

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
            model_inputs[idx + 1] = model_inputs[idx + 1][:, :, :-1, :]

        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("generated_text=", generated_text)
