# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
import forge.optimizers
from forge.verify.verify import verify, verify_backward
from test.mlir.llama.utils.utils import load_model


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
@pytest.mark.push
def test_llama_lora_fwd_pass(forge_property_recorder, model_path):
    if model_path == "openlm-research/open_llama_3b":
        pytest.skip("Insufficient host DRAM to run this model")

    # Load Model and Tokenizer for LoRA training
    use_lora = True
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora)

    # Need input seq divisible by 32 due to metal constraints TILE_WIDTH=32
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/17714 resolved
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 32

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # Compile the model for forward pass
    compiled_model = forge.compile(framework_model, input_ids, forge_property_handler=forge_property_recorder)

    verify([input_ids], framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
@pytest.mark.push
def test_llama_lora_bwd_pass(forge_property_recorder, model_path):
    if model_path == "openlm-research/open_llama_3b":
        pytest.skip("Insufficient host DRAM to run this model")

    # Load Model and Tokenizer for LoRA training
    use_lora = True
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora)
    framework_model.train()

    # Need input seq divisible by 32 due to metal constraints TILE_WIDTH=32
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/17714 resolved
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 32

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # Create a batch of input_ids
    batch_size = 32
    input_ids = input_ids.repeat(batch_size, 1)

    # Compile the model for training
    compiled_model = forge.compile(
        framework_model, input_ids, training=True, forge_property_handler=forge_property_recorder
    )

    fw_out, co_out = verify(
        [input_ids], framework_model, compiled_model, forge_property_handler=forge_property_recorder
    )

    # Run bwd pass
    grad = torch.rand_like(fw_out[0])

    verify_backward(
        [input_ids],
        grad,
        fw_out[0],
        co_out[0],
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99)),
    )
