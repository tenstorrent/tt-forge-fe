# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import forge
from test.mlir.llama.utils.utils import load_model


# TODO(tt-mlir issue #1503): This test is failing because the embedding op doesn't work with FP32.
# It should be fixed in the tt-mlir compiler soon.
@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b"])
@pytest.mark.xfail()
def test_llama_backward(model_path):
    # Load Model and Tokenizer
    framework_model, tokenizer = load_model(model_path)

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    loss_fn = torch.nn.CrossEntropyLoss()
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=1e-3)

    # Compile the model with loss and optimizer, this will invoke an autograd pass which produces bwd graph.
    compiled_model = forge.compile(
        framework_model, input_ids, loss=loss_fn, optimizer=framework_optimizer
    )
