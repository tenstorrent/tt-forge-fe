# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import requests
from PIL import Image
from datasets import load_dataset

from transformers import AutoFeatureExtractor, ViTForImageClassification

import forge
from test.utils import download_model


def generate_model_deit_imgcls_hf_pytorch(variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    image_processor = download_model(AutoFeatureExtractor.from_pretrained, variant)
    model = download_model(ViTForImageClassification.from_pretrained, variant)

    # STEP 3: Run inference on Tenstorrent device
    dataset = load_dataset("huggingface/cats-image")
    image_1 = dataset["test"]["image"][0]
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_2 = Image.open(requests.get(url, stream=True).raw)
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    # output = model(img_tensor).logits

    return model, [img_tensor], {}, compiler_cfg


variants = [
    "facebook/deit-base-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/deit-tiny-patch16-224",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_base_classify_224_hf_pytorch(variant, test_device):
    model, inputs, _, compiler_cfg = generate_model_deit_imgcls_hf_pytorch(
        variant,
    )
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_")), compiler_cfg=compiler_cfg
    )
