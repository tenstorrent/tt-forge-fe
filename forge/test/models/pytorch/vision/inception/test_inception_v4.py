# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Inception V4
import pytest
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.inception.utils.model_utils import (
    get_image,
    preprocess_timm_model,
)
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


def generate_model_inceptionV4_imgcls_osmr_pytorch(variant):
    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)

    # Load and pre-process image
    img_tensor = get_image()

    return framework_model, [img_tensor]


@pytest.mark.nightly
def test_inception_v4_osmr_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="inception", variant="v4", source=Source.OSMR, task=Task.IMAGE_CLASSIFICATION
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs = generate_model_inceptionV4_imgcls_osmr_pytorch("inceptionv4")

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_inceptionV4_imgcls_timm_pytorch(variant):
    # Load model & Preprocess image
    framework_model, img_tensor = download_model(preprocess_timm_model, variant)
    return framework_model, [img_tensor]


variants = ["inception_v4", "inception_v4.tf_in1k"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_inception_v4_timm_pytorch(record_forge_property, variant):
    if variant != "inception_v4.tf_in1k":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="inception",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs = generate_model_inceptionV4_imgcls_timm_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
