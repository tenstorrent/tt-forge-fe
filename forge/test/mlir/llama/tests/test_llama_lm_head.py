# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.verify.verify import verify


@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"])
@pytest.mark.push
def test_llama_lm_head(model_path):
    if model_path == "meta-llama/Llama-3.2-1B":
        pytest.skip("Skipping test for Llama-3.2-1B model, waiting for new transformers version.")

    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    framework_model = framework_model.lm_head
    input_features = framework_model.in_features

    # Input samples
    inputs = [
        torch.rand((1, 12, input_features)),  # Hidden states
    ]

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs=inputs, compiled_model=compiled_model, framework_model=framework_model)
