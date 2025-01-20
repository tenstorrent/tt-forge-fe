# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# BlazePose Demo Script - PyTorch

import cv2
import pytest
import torch

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name

# sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
# from mediapipepytorch.blazebase import denormalize_detections, resize_pad
# from mediapipepytorch.blazepose import BlazePose
# from mediapipepytorch.blazepalm import BlazePalm
# from mediapipepytorch.blazehand_landmark import BlazeHandLandmark
# from mediapipepytorch.blazepose_landmark import BlazePoseLandmark
# from mediapipepytorch.visualization import POSE_CONNECTIONS, draw_landmarks


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blazepose_detector_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="blazepose", variant="detector")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Load BlazePose Detector
    framework_model = BlazePose()
    framework_model.load_weights("mediapipepytorch/blazepose.pth")
    framework_model.load_anchors("mediapipepytorch/anchors_pose.npy")

    # Load data sample
    orig_image = cv2.imread("files/samples/girl.png")

    # Preprocess for BlazePose Detector
    _, img2, scale, pad = resize_pad(orig_image)
    img2 = torch.from_numpy(img2).permute((2, 0, 1)).unsqueeze(0)
    img2 = img2.float() / 255.0

    inputs = [img2]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blazepose_regressor_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="blazepose", variant="regressor")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Load BlazePose Landmark Regressor
    framework_model = BlazePoseLandmark()
    framework_model.load_weights("mediapipepytorch/blazepose_landmark.pth")

    inputs = [torch.rand(1, 3, 256, 256)]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blaze_palm_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="blazepose", variant="palm")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Load BlazePalm Detector
    framework_model = BlazePalm()
    framework_model.load_weights("mediapipepytorch/blazepalm.pth")
    framework_model.load_anchors("mediapipepytorch/anchors_palm.npy")
    framework_model.min_score_thresh = 0.75

    # Load data sample
    orig_image = cv2.imread("files/samples/girl.png")

    # Preprocess for BlazePose Detector
    img1, img2, scale, pad = resize_pad(orig_image)
    img2 = torch.from_numpy(img2).permute((2, 0, 1)).unsqueeze(0)
    img2 = img2.float() / 255.0

    inputs = [img2]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blaze_hand_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="blazepose", variant="hand")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Load BlazePalm Detector
    framework_model = BlazeHandLandmark()
    framework_model.load_weights("mediapipepytorch/blazehand_landmark.pth")

    inputs = [torch.rand(1, 3, 256, 256)]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
