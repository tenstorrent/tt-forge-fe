# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import py
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.verify.verify import verify


@pytest.mark.push
@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b"])
def test_llama_3b_lm_head(model_path):
    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    framework_model = framework_model.model.norm
    input_features = framework_model.weight.shape[0]

    # Input samples
    inputs = [
        torch.rand((1, 12, input_features)),  # Hidden states
    ]

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
@pytest.mark.skip(reason="Skipping test for Llama-3.2-1B model, waiting for new transformers version.")
@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B"])
def test_llama_32_lm_head(model_path):
    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    framework_model = framework_model.model.norm
    input_features = framework_model.weight.shape[0]

    # Input samples
    inputs = [
        torch.rand((1, 12, input_features)),  # Hidden states
    ]

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
