# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model, load_llama32
from forge.op.eval.common import compare_with_golden_pcc


@pytest.mark.parametrize("llama_ver", ["llama 3B", "llama 3.2 1B"])
@pytest.mark.xfail(reason="Waiting for the transformers version to be upgraded")
def test_llama_mlp(llama_ver):
    # Load Llama model and tokenizer
    if llama_ver == "llama 3B":
        framework_model, _ = load_model()
    elif llama_ver == "llama 3.2 1B":
        framework_model, _ = load_llama32()
    framework_model = framework_model.model.layers[0].mlp
    hidden_dim = framework_model.hidden_size

    # Input samples
    inputs = [
        torch.rand((1, 12, hidden_dim)),  # Hidden states
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
