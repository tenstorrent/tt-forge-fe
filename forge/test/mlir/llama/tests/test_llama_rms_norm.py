# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.mlir.llama.utils.utils import load_model

import pytest
import torch

import forge
from forge.verify.config import VerifyConfig
from forge.verify.verify import verify


@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"])
@pytest.mark.push
def test_llama_lm_head(model_path):
    if model_path == "meta-llama/Llama-3.2-1B":
        pytest.skip("Skipping test for Llama-3.2-1B model, waiting for new transformers version.")

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
