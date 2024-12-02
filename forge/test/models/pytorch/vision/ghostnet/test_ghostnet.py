# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import urllib
from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from test.utils import download_model

import forge

variants = ["ghostnet_100"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_ghostnet_timm(variant, test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename)
    data_config = resolve_data_config({}, model=framework_model)
    transforms = create_transform(**data_config, is_training=False)
    img_tensor = transforms(img).unsqueeze(0)

    compiled_model = forge.compile(framework_model, sample_inputs=[img_tensor], module_name=f"pt_{variant}")
