# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import forge
from forge.op.eval.common import compare_with_golden_pcc
from test.mlir.llama.utils.utils import load_model


@pytest.mark.xfail()
def test_llama_prefill_on_cpu_decode_on_tt_no_cache():

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model, tokenizer = load_model(model_path=model_path, use_cache=False)
    tokenizer.pad_token_id = framework_model.config.pad_token_id

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
    next_token_logits = logits[0][:, non_padding_seq_len - 1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    # Update the input_ids with predicted token from prefill in the first padding token index
    input_ids[:, non_padding_seq_len] = next_token
    attention_mask[:, non_padding_seq_len] = 1

    non_padding_seq_len = int(torch.sum(attention_mask))
    padding_seq_len = attention_mask.shape[1] - non_padding_seq_len

    # # Compile the model on TT
    # compiled_model = forge.compile(
    #     framework_model, sample_inputs=[input_ids, attention_mask]
    # )

    # Run decode stage on TT device and generate tokens by appending predicted token into sequence of input tokens
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    max_new_tokens = padding_seq_len
    for idx in range(max_new_tokens):

        model_inputs = [input_ids, attention_mask]

        # CPU Inference
        framework_output = framework_model(input_ids=input_ids, attention_mask=attention_mask)

        # # Run on TT device
        # tt_output = compiled_model(*model_inputs)
        # tt_output = [tt_out.to("cpu") for tt_out in tt_output]

        # # Validate TT result with Framework
        # assert all(
        #     [
        #         compare_with_golden_pcc(golden=fw_out, calculated=tt_out, pcc=0.99)
        #         for fw_out, tt_out in zip(framework_output, tt_output)
        #     ]
        # )

        next_token_index = non_padding_seq_len + idx
        next_token_logits = framework_output[0][:, next_token_index - 1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        input_ids[:, next_token_index] = next_token
        attention_mask[:, next_token_index] = 1

    # Generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("generated_text=", generated_text)


@pytest.mark.xfail()
def test_llama_prefill_on_cpu_decode_on_tt_cache():

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model, tokenizer = load_model(model_path=model_path, use_cache=True, return_dict=True)

    class LlamaModelWrapper(torch.nn.Module):
        """
        In LlamaModelWrapper class, forward function takes single input token (i.e last predicted token), attention_mask, position_ids
        and past key values from the previous iteration and return logits and past key values
        Args:
            input_id (`torch.Tensor`) - shape of (batch_size, 1)
            past_key_values (`List[List[torch.Tensor, torch.Tensor]]`) - key/values shape - (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)
            attention_mask (`torch.Tensor`) -  shape of (batch_size, key_value_seq_len + 1)
            position_ids (`torch.Tensor`) -  shape of (batch_size, 1)
        Returns:
            outputs - Logits of shape (batch_size, 1, vocab_size) and past_key_values (`List[List[torch.Tensor, torch.Tensor]]`)
            past key/values tensor shape - (batch_size, num_of_key_values_heads, key_value_seq_len + 1, head_dim)
        """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_id, attention_mask, position_ids, past_key_values):
            model_outputs = self.model(
                input_ids=input_id,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
            return model_outputs.logits, model_outputs.past_key_values

    llama_model = LlamaModelWrapper(framework_model)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run Prefill on CPU with cache to get the initial logits and past key-values
    prefill_output = framework_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    next_token_logits = prefill_output.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    generated_tokens = inputs.input_ids
    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    past_key_values_list = [[k, v] for k, v in prefill_output.past_key_values]

    model_inputs = [next_token.unsqueeze(0), past_key_values_list]

    # Padding on past key values can be enabled by setting apply_padding_on_past_key_values to True.
    apply_padding_on_past_key_values = True
    max_new_tokens = 46
    if apply_padding_on_past_key_values:
        padding_seq_len = max_new_tokens
        non_padding_seq_len = past_key_values_list[0][0].shape[-2]

        # Zero Pad past key values in key_value_seq_len(i.e -2) dimension
        # Before padding past key values tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)
        # After Padding Past key value tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len + padding_seq_len, head_dim)
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

    # # Compile the model
    # compiled_model = forge.compile(llama_model, sample_inputs=model_inputs)

    # Run decode stage on TT device and generate tokens by passing the last predicted token and the past key values.
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    attention_mask = None
    position_ids = None
    for max_new_tokens_idx in range(max_new_tokens):

        non_padding_past_key_value_seq_length = non_padding_seq_len + max_new_tokens_idx
        if apply_padding_on_past_key_values:
            padded_past_key_seq_length = model_inputs[1][0][0].shape[2]
            input_seq_len = model_inputs[0].shape[-1]

            # Calculate attention mask
            attention_mask = torch.zeros(padded_past_key_seq_length + input_seq_len)
            attention_mask[:non_padding_past_key_value_seq_length] = 1
            attention_mask[-1] = 1
            attention_mask = attention_mask.unsqueeze(0)

            # Calculate position ids
            position_ids = torch.arange(
                non_padding_past_key_value_seq_length,
                input_seq_len + non_padding_past_key_value_seq_length,
                dtype=torch.long,
            )
            position_ids = position_ids.unsqueeze(0)

        # CPU Inference
        model_outputs = llama_model(model_inputs[0], attention_mask, position_ids, model_inputs[1])

        # TT will return the logits and past key values as list of tensor, so flattening
        # framework output past key values from List(List(Key1, Values1), ... , List(Key26, Values26)) to
        # List(Key1, Values1, ... , Key26, Values26) for comparing the Framework and TT output in similar fashion.
        framework_output = [model_outputs[0]]
        for k, v in model_outputs[1]:
            framework_output.append(k)
            framework_output.append(v)

        # # Run on TT device
        # tt_output = compiled_model(*model_inputs)
        # tt_output = [tt_out.to("cpu") for tt_out in tt_output]

        # # Validate TT result with Framework
        # assert all([compare_with_golden_pcc(golden=fw_out, calculated=tt_out, pcc=0.99) for fw_out, tt_out in zip(framework_output, tt_output)])

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

        if apply_padding_on_past_key_values:

            for idx in range(len(model_inputs[1])):
                model_inputs[1][idx][0][:, :, non_padding_past_key_value_seq_length, :] = model_inputs[1][idx][0][
                    :, :, -1, :
                ]
                model_inputs[1][idx][0] = model_inputs[1][idx][0][:, :, :-1, :]
                model_inputs[1][idx][1][:, :, non_padding_past_key_value_seq_length, :] = model_inputs[1][idx][1][
                    :, :, -1, :
                ]
                model_inputs[1][idx][1] = model_inputs[1][idx][1][:, :, :-1, :]

        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("generated_text=", generated_text)
