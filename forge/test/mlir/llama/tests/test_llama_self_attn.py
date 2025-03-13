# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig


@pytest.mark.parametrize(
    "model_path",
    [
        "openlm-research/open_llama_3b",
        "meta-llama/Llama-3.2-1B",
    ],
)
@pytest.mark.push
def test_llama_self_attn(model_path):
    # Define wrapper function
    class SelfAttention(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *inputs):
            hidden_states, _, _ = self.model(*inputs)

            return hidden_states

    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)
    framework_model = SelfAttention(framework_model.model.layers[0].self_attn)

    # Get hidden dimension
    hidden_size = framework_model.model.config.hidden_size

    # Input samples
    inputs = [
        torch.rand((1, 12, hidden_size)),  # Hidden states
        torch.ones((1, 1, 12, 12)),  # Attention mask
        torch.arange(12).unsqueeze(0).float(),  # Position IDs
    ]

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
