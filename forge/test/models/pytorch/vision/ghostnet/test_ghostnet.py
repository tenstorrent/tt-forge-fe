# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib

import pytest
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, build_module_name
from test.utils import download_model

variants = ["ghostnet_100"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_ghostnet_timm(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="ghostnet", variant=variant, source=Source.TIMM)

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

    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
