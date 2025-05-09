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

from test.models.pytorch.vision.wideresnet.utils.utils import (
    generate_model_wideresnet_imgcls_pytorch,
    post_processing,
)
from test.utils import download_model

variants = [
    pytest.param("wide_resnet50_2", marks=[pytest.mark.push]),
    pytest.param("wide_resnet101_2"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_wideresnet_pytorch(forge_property_recorder, variant):
    if variant != "wide_resnet50_2":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="wideresnet",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and prepare input data
    (framework_model, inputs) = generate_model_wideresnet_imgcls_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post processing
    post_processing(co_out)


def generate_model_wideresnet_imgcls_timm(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return framework_model, [img_tensor]


variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_wideresnet_timm(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="wideresnet",
        source=Source.TIMM,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    (framework_model, inputs) = generate_model_wideresnet_imgcls_timm(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
