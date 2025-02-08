# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from test.mlir.llama.utils.utils import load_model


@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b"])
def test_llama_backward(model_path):
    # Load Model and Tokenizer
    framework_model, tokenizer = load_model(model_path)

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Compile the model with loss and optimizer, this will invoke an autograd pass which produces bwd graph.
    compiled_model = forge.compile(framework_model, input_ids, training=True)
