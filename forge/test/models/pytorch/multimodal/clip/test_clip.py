# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.multimodal.clip.model_utils.clip_model import CLIPTextWrapper
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "openai/clip-vit-base-patch32",
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_clip_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.CLIP,
        variant=variant,
        suffix="text",
        source=Source.HUGGINGFACE,
        task=Task.TEXT_GENERATION,
    )

    # Load processor and model from HuggingFace
    model = download_model(CLIPModel.from_pretrained, variant, torchscript=True)
    processor = download_model(CLIPProcessor.from_pretrained, variant)

    # Load image from the IAM dataset
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Process image
    text = [
        "a photo of a cat",
        "a photo of a dog",
    ]
    inputs = processor(text=text, images=image, return_tensors="pt")

    inputs = [inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]]
    framework_model = CLIPTextWrapper(model)
    inputs = [inputs[0], inputs[2]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
