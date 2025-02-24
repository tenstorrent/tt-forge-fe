# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.verify.compare import compare_with_golden

from test.models.pytorch.multimodal.deepseek.utils.model import (
    DeepSeekWrapper_decoder,
    download_model_and_tokenizer,
)


def prefill_on_cpu(model, input_ids):
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


@pytest.mark.parametrize("variant", ["deepseek-math-7b-instruct"])
def test_deepseek_prefil_on_device_decode_on_cpu(variant):
    """
    This function tests the inference of the deepseek_math model split into two parts:
    - The first part is the prefilling of the model on the device.
    - The second part is the decoding of the model on the CPU without KV cache.
    """

    model_name = f"deepseek-ai/{variant}"
    model, tokenizer, input_ids = download_model_and_tokenizer(model_name)

    input_ids = input_ids.to(torch.int32)

    # This is the part of the model needed for prefill; model without the last Linear layer (lm_head)
    model_decoder = model.get_decoder()
    model_decoder = DeepSeekWrapper_decoder(model_decoder)
    compiled_decoder = forge.compile(model_decoder, sample_inputs=input_ids)

    # Prefill Phase - Process the initial prompt on device
    transformer_outputs = compiled_decoder(input_ids)
    # Get hidden states for all tokens from the last "transformer layer".
    hidden_states_compiled = transformer_outputs[0]

    # Get hidden states for all tokens from the last "transformer layer" calculated on CPU.
    hidden_states_framework = prefill_on_cpu(model, input_ids)

    # Compare result of prefilling on device with the result of prefilling on CPU.
    # Calculate the pcc for only the last vector in the hidden states tensor.
    assert compare_with_golden(hidden_states_framework[:, -1, :], hidden_states_compiled[:, -1, :])
    # Decode Phase - Generate new tokens
    max_new_tokens = 200
    output_ids_compiled, output_logits_compiled = decode_on_cpu(
        model, tokenizer, input_ids, hidden_states_compiled, max_new_tokens
    )
    output_ids_framework, output_logits_framework = decode_on_cpu(
        model, tokenizer, input_ids, hidden_states_framework, max_new_tokens
    )

    assert compare_with_golden(golden=output_logits_framework[0], calculated=output_logits_compiled[0])

    # Generated text
    generated_text_compiled = tokenizer.decode(output_ids_compiled[0], skip_special_tokens=True)
    print(f"generated_text_compiled = {generated_text_compiled}")
