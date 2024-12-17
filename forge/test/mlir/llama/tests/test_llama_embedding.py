# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from forge.verify.verify import verify


@pytest.mark.push
@pytest.mark.xfail()
@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b"])
def test_llama_3b_embedding(model_path):
    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    vocab_size = framework_model.config.vocab_size
    framework_model = framework_model.model.embed_tokens

    # Input samples
    inputs = [
        torch.randint(0, vocab_size, (1, 12)),  # Input token IDs
    ]

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B"])
def test_llama_32_embedding(model_path):
    # Load Llama model and tokenizer
    framework_model, _ = load_model(model_path)

    vocab_size = framework_model.config.vocab_size
    framework_model = framework_model.model.embed_tokens

    # Input samples
    inputs = [
        torch.randint(0, vocab_size, (1, 12)),  # Input token IDs
    ]

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
