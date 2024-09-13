# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.op.eval.common import compare_with_golden_pcc


@pytest.mark.xfail()
def test_llama_self_attn():
    # Define wrapper function
    class SelfAttention(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *inputs):
            hidden_states, _, _ = self.model(*inputs)

            return hidden_states
    
    # Load Llama 3B model and tokenizer
    framework_model = load_model()
    framework_model = SelfAttention(framework_model.model.layers[0].self_attn)

    # Input samples
    inputs = [
        torch.rand((1, 12, 3200)), # Hidden states
        torch.ones((1, 1, 12, 12)), # Attention mask
        torch.arange(12).unsqueeze(0), # Position IDs
    ]

    # Sanity run
    golden_output = framework_model(*inputs)

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    
    # Run on TT device
    tt_out = compiled_model(*inputs)
    tt_out = [out.to("cpu") for out in tt_out]
    
    # Validate results
    assert compare_with_golden_pcc(golden=golden_output, calculated=tt_out[0], pcc=0.99)
