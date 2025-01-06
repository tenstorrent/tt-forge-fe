# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.openpose.utils.model import (
    OpenPoseBodyModel,
    OpenPoseHandModel,
    get_image_tensor,
    transfer,
)
from test.models.utils import Framework, Source, build_module_name
from test.utils import download_model

variants = [
    "body_basic",
    "hand_basic",
]


def generate_model_openpose_posdet_custom_pytorch(variant):
    # Load model
    if variant == "body_basic":
        model_path = "weights/body_pose_model.pth"
        framework_model = OpenPoseBodyModel()
        sample_path = "samples/body.jpeg"

    elif variant == "hand_basic":
        model_path = "weights/hand_pose_model.pth"
        framework_model = OpenPoseHandModel()
        sample_path = "samples/hand.jpeg"
    framework_model_dict = transfer(framework_model, torch.load(model_path))
    framework_model.load_state_dict(framework_model_dict)

    # Load & pre-process image
    img_tensor = get_image_tensor(sample_path)

    return framework_model, [img_tensor], {}


@pytest.mark.parametrize("variant", variants)
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_openpose_basic(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="openpose", variant=variant, suffix="basic")

    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_openpose_posdet_custom_pytorch(
        variant,
    )

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_openpose_posdet_osmr_pytorch(variant):
    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()
    # Load & pre-process image
    sample_path = "samples/body.jpeg"
    img_tensor = get_image_tensor(sample_path)

    return framework_model, [img_tensor], {}


variants = [
    "lwopenpose2d_mobilenet_cmupan_coco",
    "lwopenpose3d_mobilenet_cmupan_coco",
]


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_openpose_osmr(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="openpose", variant=variant, source=Source.OSMR)

    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_openpose_posdet_osmr_pytorch(
        variant,
    )
    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
