# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.op.eval.common import compare_with_golden_pcc


@pytest.mark.xfail(reason="L1 allocation issue on Metal")
def test_llama_embedding():    
    # Load Llama 3B model and tokenizer
    framework_model = load_model()
    vocab_size = framework_model.config.vocab_size
    framework_model = framework_model.model.embed_tokens
    
    # Input samples
    inputs = [
        torch.randint(0, vocab_size, (1, 12)), # Input token IDs
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
