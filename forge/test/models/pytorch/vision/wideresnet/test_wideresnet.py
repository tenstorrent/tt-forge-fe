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

from test.models.pytorch.vision.wideresnet.utils.utils import (
    generate_model_wideresnet_imgcls_pytorch,
    post_processing,
)
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = [
    pytest.param("wide_resnet50_2", marks=[pytest.mark.push]),
    pytest.param("wide_resnet101_2"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_wideresnet_pytorch(record_forge_property, variant):
    if variant != "wide_resnet50_2":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="wideresnet",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load the model and prepare input data
    (framework_model, inputs) = generate_model_wideresnet_imgcls_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    # Inference
    output = compiled_model(*inputs)

    # Post processing
    post_processing(output)


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
def test_wideresnet_timm(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="wideresnet",
        source=Source.TIMM,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    (framework_model, inputs) = generate_model_wideresnet_imgcls_timm(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
