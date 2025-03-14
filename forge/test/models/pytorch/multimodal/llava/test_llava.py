# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

import forge
from forge.verify.verify import verify

from .utils import load_inputs
from test.models.utils import Framework, Source, Task, build_module_name


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


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_llava(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 30 GB)")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="llava",
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, processor = load_model(variant)
    image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    text = "What’s shown in this image?"

    # Input sample
    input_ids, attn_mask, pixel_values = load_inputs(image, text, processor)
    inputs = [input_ids, attn_mask, pixel_values]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
