# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig


@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"])
@pytest.mark.push
@pytest.mark.functional
def test_llama_mlp(forge_property_recorder, model_path):
    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    framework_model = framework_model.model.layers[0].mlp
    hidden_dim = framework_model.hidden_size

    # Input samples
    inputs = [
        torch.rand((1, 12, hidden_dim)),  # Hidden states
    ]

    # Compile the model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
