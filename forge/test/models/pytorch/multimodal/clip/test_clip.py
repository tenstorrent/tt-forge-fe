# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import forge
from forge.verify.verify import verify

from test.models.pytorch.multimodal.clip.utils.clip_model import CLIPTextWrapper
from test.models.utils import Framework, build_module_name
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["openai/clip-vit-base-patch32"])
def test_clip_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="clip", variant=variant, suffix="text")

    # Record Forge Property
    record_forge_property("model_name", module_name)

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
