# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, FalconForCausalLM
import forge
from test.models.utils import build_module_name, Framework, Task


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_falcon(record_forge_property):
    variant = "tiiuae/falcon-7b-instruct"

    module_name = build_module_name(framework=Framework.PYTORCH, model="falcon", variant=variant)

    record_forge_property("module_name", module_name)

    tokenizer = AutoTokenizer.from_pretrained(variant)
    model = FalconForCausalLM.from_pretrained(variant)
    model.config.use_cache = False
    model.config.return_dict = False

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    model = Wrapper(model)
    input_tokens = tokenizer("Hello, my dog is cute", return_tensors="pt")

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # sanity
    output = model(input_tokens["input_ids"], input_tokens["attention_mask"])

    # Forge inference
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
