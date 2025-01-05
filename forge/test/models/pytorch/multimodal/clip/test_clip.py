# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import forge
import requests
import pytest
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from test.models.pytorch.multimodal.clip.utils.clip_model import (
    CLIPVisionWrapper,
    CLIPTextWrapper,
    CLIPPostProcessingWrapper,
)
import os
from test.models.utils import build_module_name, Framework, Task, Source


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_clip_pytorch(record_forge_property):
    model_ckpt = "openai/clip-vit-base-patch32"

    module_name = build_module_name(framework=Framework.PYTORCH, model="clip", variant=model_ckpt, suffix="text")

    record_forge_property("module_name", module_name)

    # Load processor and model from HuggingFace
    model = download_model(CLIPModel.from_pretrained, model_ckpt, torchscript=True)
    processor = download_model(CLIPProcessor.from_pretrained, model_ckpt)

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
    text_model = CLIPTextWrapper(model)
    inputs = [inputs[0], inputs[2]]

    compiled_model = forge.compile(text_model, sample_inputs=inputs, module_name=module_name)
