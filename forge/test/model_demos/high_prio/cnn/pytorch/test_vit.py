# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model

import os

import forge
import requests
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification


dataset = load_dataset("huggingface/cats-image")
image_1 = dataset["test"]["image"][0]
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_2 = Image.open(requests.get(url, stream=True).raw)


def generate_model_vit_imgcls_hf_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    image_processor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(ViTForImageClassification.from_pretrained, variant)
    # STEP 3: Run inference on Tenstorrent device
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    # output = model(img_tensor).logits

    return model, [img_tensor], {}


variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classify_224_hf_pytorch(variant, test_device):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE
    model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(
        test_device,
        variant,
    )

    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])


variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Redundant, already tested with test_vit_classification_1x1_demo")
def test_vit_classify_224_hf_pytorch_1x1(variant, test_device):
    os.environ["FORGE_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    if "large" in variant:
        os.environ["FORGE_EXTRA_L1_MARGIN"] = "20000"

    model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])


modes = ["verify", "demo"]
variants = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
]


@pytest.mark.skip(reason="1x1 grid size not supported yet")
@pytest.mark.parametrize("mode", modes, ids=modes)
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classification_1x1_demo(test_device, mode, variant):

    # Setup for 1x1 grid
    os.environ["FORGE_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    compiler_cfg.enable_tvm_cpu_fallback = False

    # Load image preprocessor and model
    image_processor = download_model(AutoImageProcessor.from_pretrained, variant)
    framework_model = download_model(ViTForImageClassification.from_pretrained, variant)
    model_name = "_".join(variant.split("/")[-1].split("-")[:2]) + f"_{mode}"

    # Load and preprocess image
    dataset = load_dataset("huggingface/cats-image")
    input_image = dataset["test"]["image"][0]
    input_image = image_processor(input_image, return_tensors="pt").pixel_values

    if mode == "verify":
        compiled_model = forge.compile(framework_model, sample_inputs=[input_image])
