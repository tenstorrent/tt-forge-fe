# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.llava.pytorch import (
    ModelLoader as ConditionalGenModelLoader,
)
from third_party.tt_forge_models.llava.pytorch import (
    ModelVariant as ConditionalGenModelVariant,
)
from transformers import AutoProcessor

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from PIL import Image


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
        output = self.model(**inputs)
        return output.logits


LLAVA_VARIANTS = [
    ConditionalGenModelVariant.LLAVA_1_5_7B,
]


@pytest.mark.crash_test
@pytest.mark.out_of_memory
@pytest.mark.nightly
# @pytest.mark.xfail
@pytest.mark.parametrize("variant", LLAVA_VARIANTS)
def test_llava(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.LLAVA,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    loader = ConditionalGenModelLoader()
    framework_model = loader.load_model()
    framework_model = Wrapper(framework_model)

    # Build prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What’s shown in this image?"}],
        }
    ]
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    text_prompt = processor.apply_chat_template(
        conversation, padding=True, add_generation_prompt=True
    )
    image = Image.open('forge/test/models/pytorch/multimodal/llava/australia.jpg')
    # Preprocess
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    # Input sample
    inputs = [input_ids, attention_mask, pixel_values]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
