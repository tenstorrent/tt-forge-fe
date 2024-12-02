# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os

import torch
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from test.utils import download_model
from test.models.pytorch.vision.openpose.utils.model import (
    OpenPoseBodyModel,
    OpenPoseHandModel,
    get_image_tensor,
    transfer,
)


variants = [
    "body_basic",
    "hand_basic",
]


def generate_model_openpose_posdet_custom_pytorch(test_device, variant):
    # Init config
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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

    # Sanity run
    cpu_out = framework_model(img_tensor)

    return framework_model, [img_tensor], {}


@pytest.mark.parametrize("variant", variants)
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_openpose_basic(variant, test_device):
    model, inputs, _ = generate_model_openpose_posdet_custom_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=f"pt_openpose_{variant}")


def generate_model_openpose_posdet_osmr_pytorch(test_device, variant):

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()
    # Load & pre-process image
    sample_path = "samples/body.jpeg"
    img_tensor = get_image_tensor(sample_path)

    # Sanity run
    cpu_out = framework_model(img_tensor)

    return framework_model, [img_tensor], {}


variants = [
    "lwopenpose2d_mobilenet_cmupan_coco",
    "lwopenpose3d_mobilenet_cmupan_coco",
]


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_openpose_osmr(variant, test_device):
    model, inputs, _ = generate_model_openpose_posdet_osmr_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=f"pt_openpose_{variant}")
