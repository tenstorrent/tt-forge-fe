# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules

# Third-party modules

# Forge modules

# Common constants
GIT_REPO_NAME = "tenstorrent/tt-forge-fe"

# Model path
MODEL_PATH = ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"]


@pytest.mark.parametrize("model_path", MODEL_PATH)
def test_llama_prefil_on_device_decode_on_cpu(model_path):

    if model_path == "openlm-research/open_llama_3b":
        pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 32 GB during compile time)")

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
    hidden_states_compiled = transformer_outputs[0]
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


def llama_prefill_benchmark():
    pass
