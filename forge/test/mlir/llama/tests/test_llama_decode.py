# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.mlir.llama.utils.utils import load_model

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import forge
from forge.verify.compare import compare_with_golden


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
            past_key_values (`List[List[torch.Tensor, torch.Tensor]]`)
                        key/values shape: (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)

    Returns:
        1) use_cache = False
            logits (`torch.Tensor`) - shape (batch_size, seq_length, vocab_size)
        2) use_cache = True
            logits (`torch.Tensor`) - shape (batch_size, 1, vocab_size)
            past_key_values (`List[List[torch.Tensor, torch.Tensor]]`)
                        key/values shape: (batch_size, num_of_key_values_heads, key_value_seq_len + 1, head_dim)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None):

        inputs_embeds = self.embed_tokens(input_ids)

        # The causal_mask calculated inside _update_causal_mask method in LlamaModel module
        # is not properly traced by torch.jit.trace while converting pytorch to relay ops in TVM PyTorch Frontend Conversion
        # which leads to pcc drops, because the causal mask helps to avoid attention calculation on padding token in LlamaAttention module
        # To overcome this jit trace issue, causal mask is calculated by using _prepare_4d_causal_attention_mask function.
        past_key_values_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
        )
        model_outputs = self.model(
            attention_mask=causal_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        if model_outputs.past_key_values is not None:
            return model_outputs.logits, model_outputs.past_key_values

        return model_outputs.logits


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


@pytest.mark.parametrize("run_on_tt_device", [False, True])
def test_llama_prefill_on_cpu_decode_on_tt_no_cache(run_on_tt_device):

    if run_on_tt_device:
        compiler_cfg = forge.config._get_global_compiler_config()
        compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    model, tokenizer = load_model(model_path, return_dict=True, use_cache=False, use_fast=False)
    framework_model = LlamaModelWrapper(model)
    framework_model.eval()
    tokenizer.pad_token_id = model.config.pad_token_id

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
    logits = framework_model(input_ids=input_ids, attention_mask=attention_mask)

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
        compiled_model = forge.compile(framework_model, sample_inputs=[input_ids, attention_mask])
        pytest.xfail("Found Unsupported operations while lowering from TTForge to TTIR in forward graph.")

    # Run decode stage on TT device and generate tokens by appending predicted token into sequence of input tokens
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    max_new_tokens = padding_seq_len
    for idx in range(max_new_tokens):

        # CPU Inference
        framework_output = framework_model(input_ids=input_ids, attention_mask=attention_mask)

        if run_on_tt_device:

            model_inputs = [input_ids, attention_mask]

            # Run on TT device
            tt_output = compiled_model(*model_inputs)
            tt_output = [tt_out.to("cpu") for tt_out in tt_output]

            framework_output = [framework_output] if isinstance(framework_output, torch.Tensor) else framework_output

            # Validate TT result with Framework
            assert all(
                [
                    compare_with_golden(golden=fw_out, calculated=tt_out)
                    for fw_out, tt_out in zip(framework_output, tt_output)
                ]
            )

            logits = tt_output[0]

        else:

            logits = framework_output

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


@pytest.mark.parametrize("run_on_tt_device", [False, True])
def test_llama_prefill_on_cpu_decode_on_tt_cache(run_on_tt_device):

    if run_on_tt_device:
        compiler_cfg = forge.config._get_global_compiler_config()
        compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    model, tokenizer = load_model(model_path, return_dict=True, use_cache=True, use_fast=False)
    framework_model = LlamaModelWrapper(model)
    framework_model.eval()
    tokenizer.pad_token_id = model.config.pad_token_id

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run Prefill on CPU with cache to get the initial logits and past key-values
    prefill_output = framework_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    next_token_logits = prefill_output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    generated_tokens = inputs.input_ids
    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    past_key_values_list = [[k, v] for k, v in prefill_output[1]]

    model_inputs = [next_token.unsqueeze(0), past_key_values_list]

    # Zero Pad past key values in key_value_seq_len(i.e -2) dimension
    # Before padding past key values tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)
    # After Padding Past key value tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len + max_new_tokens, head_dim)
    max_new_tokens = 46
    non_padding_seq_length = past_key_values_list[0][0].shape[-2]
    for idx, (k, v) in enumerate(model_inputs[1]):
        model_inputs[1][idx][0] = torch.cat(
            [
                k,
                torch.zeros(k.shape[-4], k.shape[-3], max_new_tokens, k.shape[-1]).to(k.dtype),
            ],
            dim=-2,
        )
        model_inputs[1][idx][1] = torch.cat(
            [
                v,
                torch.zeros(v.shape[-4], v.shape[-3], max_new_tokens, v.shape[-1]).to(k.dtype),
            ],
            dim=-2,
        )

    if run_on_tt_device:
        # Calculate attention mask and postion_ids
        padded_past_key_values_seq_length = model_inputs[1][0][0].shape[-2]
        input_seq_length = model_inputs[0].shape[-1]
        attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
            padded_past_key_values_seq_length, non_padding_seq_length, input_seq_length
        )

        # Compile the model
        compiled_model = forge.compile(
            framework_model, sample_inputs=[model_inputs[0], attention_mask, position_ids, model_inputs[1]]
        )
        pytest.xfail("Found Unsupported operations while lowering from TTForge to TTIR in forward graph.")

    # Run decode stage on TT device and generate tokens by passing the last predicted token and the past key values.
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    for max_new_tokens_idx in range(max_new_tokens):

        non_padding_past_key_values_seq_length = non_padding_seq_length + max_new_tokens_idx
        padded_past_key_values_seq_length = model_inputs[1][0][0].shape[-2]
        input_seq_length = model_inputs[0].shape[-1]
        attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
            padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
        )

        # CPU Inference
        model_outputs = framework_model(model_inputs[0], attention_mask, position_ids, model_inputs[1])

        # TT will return the logits and past key values as list of tensor, so flattening
        # framework output past key values from List(List(Key1, Values1), ... , List(Key26, Values26)) to
        # List(Key1, Values1, ... , Key26, Values26) for comparing the Framework and TT output in similar fashion.
        framework_output = [model_outputs[0]]
        for k, v in model_outputs[1]:
            framework_output.append(k)
            framework_output.append(v)

        if run_on_tt_device:
            # Run on TT device
            tt_inputs = [model_inputs[0], attention_mask, position_ids, model_inputs[1]]
            tt_output = compiled_model(*tt_inputs)
            tt_output = [tt_out.to("cpu") for tt_out in tt_output]

            # Validate TT result with Framework
            assert all(
                [
                    compare_with_golden(golden=fw_out, calculated=tt_out)
                    for fw_out, tt_out in zip(framework_output, tt_output)
                ]
            )

            logits = tt_output[0]
            past_key_values = tt_output[1:]

        else:
            logits = framework_output[0]
            past_key_values = framework_output[1:]

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        model_inputs = [next_token.unsqueeze(0)]

        # Formate the past key values from List(Key1, Values1, ... , Key26, Values26) to
        # List(List(Key1, Values1), ... , List(Key26, Values26))
        model_inputs.append([past_key_values[idx : idx + 2] for idx in range(0, len(past_key_values), 2)])

        # Generated new token, key and values states are appended to the past key values tensor on key_values_sequence_length dim (i.e -2).
        # For predicting next token, move the generated new token, key and values states from the last past key values tensor index
        # to the first past key values tensor padding index on key_values_sequence_length dim (i.e -2).
        for idx in range(len(model_inputs[1])):
            model_inputs[1][idx][0][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[1][idx][0][
                :, :, -1, :
            ]
            model_inputs[1][idx][0] = model_inputs[1][idx][0][:, :, :-1, :]
            model_inputs[1][idx][1][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[1][idx][1][
                :, :, -1, :
            ]
            model_inputs[1][idx][1] = model_inputs[1][idx][1][:, :, :-1, :]

        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("generated_text=", generated_text)
