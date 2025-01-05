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
import torch
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name, Framework

variants = ["ghostnet_100"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_ghostnet_timm(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="ghostnet", variant=variant, source="timm")

    record_forge_property("module_name", module_name)

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

    compiled_model = forge.compile(framework_model, sample_inputs=[img_tensor], module_name=module_name)
    co_out = compiled_model(img_tensor)
    fw_out = framework_model(img_tensor)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
