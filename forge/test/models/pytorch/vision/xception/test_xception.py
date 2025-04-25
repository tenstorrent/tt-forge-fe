# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib

import pytest
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.xception.utils.utils import post_processing
from test.utils import download_model


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


params = [
    pytest.param(
        "xception",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("xception41"),
    pytest.param("xception65"),
    pytest.param("xception71"),
    pytest.param("xception71.tf_in1k", marks=[pytest.mark.push, pytest.mark.models]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", params)
def test_xception_timm(forge_property_recorder, variant):
    if variant not in ["xception", "xception71.tf_in1k"]:
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="xception",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    (framework_model, inputs) = generate_model_xception_imgcls_timm(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post Processing
    if variant == "xception71.tf_in1k":
        post_processing(co_out)
