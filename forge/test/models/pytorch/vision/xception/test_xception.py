# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
import urllib
from test.utils import download_model
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os


def generate_model_xception_imgcls_timm(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return framework_model, [img_tensor], compiler_cfg


variants = ["xception", "xception41", "xception65", "xception71"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xception_timm(variant, test_device):

    (model, inputs, compiler_cfg) = generate_model_xception_imgcls_timm(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{variant}_timm", compiler_cfg=compiler_cfg)
