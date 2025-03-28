# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.vovnet.utils.model_utils import (
    get_image,
    preprocess_steps,
    preprocess_timm_model,
)
from test.models.pytorch.vision.vovnet.utils.src_vovnet_stigma import vovnet39, vovnet57
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


def generate_model_vovnet_imgcls_osmr_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    image_tensor = get_image()

    return model, [image_tensor], {}


varaints = [
    "vovnet27s",
    "vovnet39",
    "vovnet57",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_vovnet_osmr_pytorch(forge_property_recorder, variant):
    if variant != "vovnet27s":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="vovnet", variant=variant, source=Source.OSMR, task=Task.OBJECT_DETECTION
    )

    # Record Forge Property
    if variant in ["vovnet27s"]:
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_vovnet_imgcls_osmr_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_vovnet39_imgcls_stigma_pytorch():
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet39)
    return model, [image_tensor], {}


@pytest.mark.nightly
def test_vovnet_v1_39_stigma_pytorch(forge_property_recorder):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    variant = "vovnet39"

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="vovnet_v1",
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_vovnet39_imgcls_stigma_pytorch()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_vovnet57_imgcls_stigma_pytorch():
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet57)

    return model, [image_tensor], {}


@pytest.mark.nightly
def test_vovnet_v1_57_stigma_pytorch(forge_property_recorder):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    variant = "vovnet_v1_57"

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="vovnet",
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_vovnet57_imgcls_stigma_pytorch()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_vovnet_imgcls_timm_pytorch(variant):
    model, image_tensor = download_model(preprocess_timm_model, variant)

    return model, [image_tensor], {}


variants = [
    "ese_vovnet19b_dw",
    "ese_vovnet39b",
    "ese_vovnet99b",
    pytest.param(
        "ese_vovnet19b_dw.ra_in1k",
        marks=[
            pytest.mark.xfail(
                reason="AssertionError: Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to broadcast: [1, 1, 7, 7168] vs [1, 1, 7, 7]"
            )
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vovnet_timm_pytorch(forge_property_recorder, variant):
    if variant != "ese_vovnet19b_dw.ra_in1k":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="vovnet",
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_vovnet_imgcls_timm_pytorch(
        variant,
    )
    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
