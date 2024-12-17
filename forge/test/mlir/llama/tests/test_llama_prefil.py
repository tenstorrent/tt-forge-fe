# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from transformers import LlamaTokenizer

import forge
from test.mlir.llama.utils.utils import load_model
from forge.verify.compare import compare_with_golden


def prefil_on_cpu(model, input_ids):
    with torch.no_grad():
        transformer_outputs = model.model(
            input_ids=input_ids,  # Pass the entire updated sequence
        )
        hidden_states = transformer_outputs.last_hidden_state
    return hidden_states


def decode_on_cpu(model, tokenizer, input_ids, hidden_states, max_new_tokens):
    output_logits = []
    for i in range(max_new_tokens):
        # Use only the hidden state of the last token in the sequence
        next_token_logits = model.lm_head(hidden_states[:, -1, :])
        output_logits.append(next_token_logits)

        # Get the next token ID
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # Stop if the EOS token is generated
        if next_token_id == tokenizer.eos_token_id:
            break

        # Update input_ids with the new token
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        # Run the model again to get the new hidden state
        # Here we do effectively prefil again because we don't use KV cache.
        # If we used KV cache, we could just run the model with for the new token.
        with torch.no_grad():
            transformer_outputs = model.model(
                input_ids=input_ids,  # Pass the entire updated sequence
            )
        hidden_states = transformer_outputs.last_hidden_state

    return input_ids, output_logits


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b"])
def test_llama_3b_prefil_on_device_decode_on_cpu(model_path):
    """
    This function tests the inference of the Llama models split into two parts:
    - The first part is the prefilling of the model on the device.
    - The second part is the decoding of the model on the CPU without KV cache.
    """

    # Load Llama model and tokenizer
    model, tokenizer = load_model(model_path, return_dict=True)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # This is the part of the model needed for prefill; model without the last Linear layer (lm_head)
    model_decoder = model.get_decoder()
    compiled_decoder = forge.compile(model_decoder, sample_inputs=input_ids)

    # Prefill Phase - Process the initial prompt on device
    transformer_outputs = compiled_decoder(input_ids)
    # Get hidden states for all tokens from the last "transformer layer".
    hidden_states_compiled = transformer_outputs.last_hidden_state
    hidden_states_compiled = hidden_states_compiled.to("cpu")

    # Get hidden states for all tokens from the last "transformer layer" calculated on CPU.
    hidden_states_framework = prefil_on_cpu(model, input_ids)

    # Compare result of prefilling on device with the result of prefilling on CPU.
    # Calculate the pcc for only the last vector in the hidden states tensor.
    assert compare_with_golden(hidden_states_framework[:, -1, :], hidden_states_compiled[:, -1, :])

    # Decode Phase - Generate new tokens
    max_new_tokens = 46
    output_ids_compiled, output_logits_compiled = decode_on_cpu(
        model, tokenizer, input_ids, hidden_states_compiled, max_new_tokens
    )
    _, output_logits_framework = decode_on_cpu(model, tokenizer, input_ids, hidden_states_framework, max_new_tokens)

    # Compare the logits of the generated tokens with the golden values from CPU.
    assert all(
        [
            compare_with_golden(golden=out_logits_fw, calculated=out_logits_tt)
            for out_logits_fw, out_logits_tt in zip(output_logits_framework, output_logits_compiled)
        ]
    )

    # Generated text
    generated_text_compiled = tokenizer.decode(output_ids_compiled[0], skip_special_tokens=True)
    print(generated_text_compiled)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B"])
def test_llama_32_prefil_on_device_decode_on_cpu(model_path):
    """
    This function tests the inference of the Llama models split into two parts:
    - The first part is the prefilling of the model on the device.
    - The second part is the decoding of the model on the CPU without KV cache.
    """

    # Load Llama model and tokenizer
    model, tokenizer = load_model(model_path, return_dict=True)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # This is the part of the model needed for prefill; model without the last Linear layer (lm_head)
    model_decoder = model.get_decoder()
    compiled_decoder = forge.compile(model_decoder, sample_inputs=input_ids)

    # Prefill Phase - Process the initial prompt on device
    transformer_outputs = compiled_decoder(input_ids)
    # Get hidden states for all tokens from the last "transformer layer".
    hidden_states_compiled = transformer_outputs.last_hidden_state
    hidden_states_compiled = hidden_states_compiled.to("cpu")

    # Get hidden states for all tokens from the last "transformer layer" calculated on CPU.
    hidden_states_framework = prefil_on_cpu(model, input_ids)

    # Compare result of prefilling on device with the result of prefilling on CPU.
    # Calculate the pcc for only the last vector in the hidden states tensor.
    assert compare_with_golden(hidden_states_framework[:, -1, :], hidden_states_compiled[:, -1, :])

    # Decode Phase - Generate new tokens
    max_new_tokens = 46
    output_ids_compiled, output_logits_compiled = decode_on_cpu(
        model, tokenizer, input_ids, hidden_states_compiled, max_new_tokens
    )
    _, output_logits_framework = decode_on_cpu(model, tokenizer, input_ids, hidden_states_framework, max_new_tokens)

    # Compare the logits of the generated tokens with the golden values from CPU.
    assert all(
        [
            compare_with_golden(golden=out_logits_fw, calculated=out_logits_tt)
            for out_logits_fw, out_logits_tt in zip(output_logits_framework, output_logits_compiled)
        ]
    )

    # Generated text
    generated_text_compiled = tokenizer.decode(output_ids_compiled[0], skip_special_tokens=True)
    print(generated_text_compiled)
