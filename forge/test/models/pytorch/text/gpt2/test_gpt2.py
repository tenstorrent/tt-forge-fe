# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


# Wrapper to get around past key values
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, None, attention_mask)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["gpt2"])
def test_gpt2_text_gen(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="gpt2", variant=variant, task=Task.TEXT_GENERATION, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load tokenizer and model from HuggingFace
    config = GPT2Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPT2Config(**config_dict)
    model = download_model(GPT2LMHeadModel.from_pretrained, variant, config=config)

    input_ids = torch.cat(
        [torch.randint(1, model.config.vocab_size, (1, 255)), torch.zeros(1, 1, dtype=torch.int64)], dim=-1
    ).to(torch.int64)
    attn_mask = torch.ones(1, 256)
    inputs = [input_ids, attn_mask]

    framework_model = Wrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["mnoukhov/gpt2-imdb-sentiment-classifier"])
def test_gpt2_sequence_classification(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="gpt2",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, padding_side="left")
    model = download_model(AutoModelForSequenceClassification.from_pretrained, variant, return_dict=False)
    model.eval()
    framework_model = Wrapper(model)

    # Prepare input
    test_input = "This is a sample text from "
    input_tokens = tokenizer(test_input, return_tensors="pt")
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
