# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.verify.verify import verify


@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"])
@pytest.mark.push
def test_llama_embedding(model_path):
    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    vocab_size = framework_model.config.vocab_size
    framework_model = framework_model.model.embed_tokens

    # Input samples
    # cast input_ids to int32 since int64 causes embedding op data mismatch. Tracking issue: https://github.com/tenstorrent/tt-forge-fe/issues/952
    inputs = [
        torch.randint(0, vocab_size, (1, 12), dtype=torch.int32),  # Input token IDs
    ]

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
