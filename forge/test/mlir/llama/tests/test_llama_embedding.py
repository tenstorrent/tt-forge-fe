# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.op.eval.common import compare_with_golden_pcc


@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"])
@pytest.mark.xfail(reason="RuntimeError: ttnn.embedding op fails while validating the input_tensor layout")
def test_llama_embedding(model_path):
    if model_path == "meta-llama/Llama-3.2-1B":
        pytest.skip("Skipping test for Llama-3.2-1B model, waiting for new transformers version.")

    # RuntimeError: ttnn.embedding op fails while validating the input_tensor layout(i,e it is in ROW_MAJOR LAYOUT)
    # tt-mlir issue: https://github.com/tenstorrent/tt-mlir/issues/679

    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    vocab_size = framework_model.config.vocab_size
    framework_model = framework_model.model.embed_tokens

    # Input samples
    inputs = [
        torch.randint(0, vocab_size, (1, 12)),  # Input token IDs
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
