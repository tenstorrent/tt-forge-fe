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


@pytest.mark.nightly
def test_clip_pytorch(test_device):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load processor and model from HuggingFace
    model_ckpt = "openai/clip-vit-base-patch32"
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

    compiled_model = forge.compile(text_model, sample_inputs=inputs, module_name="pt_clip_text_model")
