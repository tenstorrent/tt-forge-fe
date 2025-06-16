# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.multimodal.llava.model_utils.utils import load_inputs


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
        output = self.model(**inputs)
        return output.logits


def load_model(variant):
    processor = AutoProcessor.from_pretrained(variant)
    model = LlavaForConditionalGeneration.from_pretrained(variant)
    model = Wrapper(model)
    return model, processor


variants = ["llava-hf/llava-1.5-7b-hf"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.xfail
def test_llava(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.LLAVA,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    raise RuntimeError("Requires multi-chip support")

    framework_model, processor = load_model(variant)
    image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    text = "What’s shown in this image?"

    # Input sample
    input_ids, attn_mask, pixel_values = load_inputs(image, text, processor)
    inputs = [input_ids, attn_mask, pixel_values]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
