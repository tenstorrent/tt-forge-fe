# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import forge
import pytest
import os
from transformers import ViTImageProcessor
import timm
from test.utils import download_model
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


@pytest.mark.nightly
def test_swin_v1_tiny_4_224_hf_pytorch(test_device):
    # pytest.skip() # Working on it
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    # model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", torchscript=True)
    model = download_model(timm.create_model, "swin_tiny_patch4_window7_224", pretrained=True)
    model.eval()

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    print(img_tensor.shape)

    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_swin_tiny_patch4_window7_224")
