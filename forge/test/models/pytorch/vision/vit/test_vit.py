# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
import requests
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
import os
from test.models.utils import build_module_name


dataset = load_dataset("huggingface/cats-image")
image_1 = dataset["test"]["image"][0]
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_2 = Image.open(requests.get(url, stream=True).raw)


def generate_model_vit_imgcls_hf_pytorch(test_device, variant):
    # STEP 2: Create Forge module from PyTorch model
    image_processor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(ViTForImageClassification.from_pretrained, variant)
    # STEP 3: Run inference on Tenstorrent device
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    # output = model(img_tensor).logits

    return model, [img_tensor], {}


variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classify_224_hf_pytorch(variant, test_device):
    model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(
        test_device,
        variant,
    )

    module_name = build_module_name(framework="pt", model="vit", variant=variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
