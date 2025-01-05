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
from test.models.utils import build_module_name, Framework, Task


def generate_model_xception_imgcls_timm(variant):
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

    return framework_model, [img_tensor]


variants = ["xception", "xception41", "xception65", "xception71"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xception_timm(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="xception", variant=variant, source="timm")

    record_forge_property("module_name", module_name)

    (model, inputs,) = generate_model_xception_imgcls_timm(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
