# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# BlazePose Demo Script - PyTorch

import pytest
import cv2
import forge
import torch
import sys
import os
from test.models.utils import build_module_name

# sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
# from mediapipepytorch.blazebase import denormalize_detections, resize_pad
# from mediapipepytorch.blazepose import BlazePose
# from mediapipepytorch.blazepalm import BlazePalm
# from mediapipepytorch.blazehand_landmark import BlazeHandLandmark
# from mediapipepytorch.blazepose_landmark import BlazePoseLandmark
# from mediapipepytorch.visualization import POSE_CONNECTIONS, draw_landmarks


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blazepose_detector_pytorch(record_forge_property):
    # Load BlazePose Detector
    pose_detector = BlazePose()
    pose_detector.load_weights("mediapipepytorch/blazepose.pth")
    pose_detector.load_anchors("mediapipepytorch/anchors_pose.npy")

    # Load data sample
    orig_image = cv2.imread("files/samples/girl.png")

    # Preprocess for BlazePose Detector
    _, img2, scale, pad = resize_pad(orig_image)
    img2 = torch.from_numpy(img2).permute((2, 0, 1)).unsqueeze(0)
    img2 = img2.float() / 255.0
    module_name = build_module_name(framework="pt", model="blazepose", task="detector")
    compiled_model = forge.compile(pose_detector, sample_inputs=[img2], module_name=module_name)


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blazepose_regressor_pytorch(record_forge_property):
    # Load BlazePose Landmark Regressor
    pose_regressor = BlazePoseLandmark()
    pose_regressor.load_weights("mediapipepytorch/blazepose_landmark.pth")
    img2 = [torch.rand(1, 3, 256, 256)]
    module_name = build_module_name(framework="pt", model="blazepose", task="regressor")
    compiled_model = forge.compile(pose_regressor, sample_inputs=img2, module_name=module_name)


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blaze_palm_pytorch(record_forge_property):
    # Load BlazePalm Detector
    palm_detector = BlazePalm()
    palm_detector.load_weights("mediapipepytorch/blazepalm.pth")
    palm_detector.load_anchors("mediapipepytorch/anchors_palm.npy")
    palm_detector.min_score_thresh = 0.75

    # Load data sample
    orig_image = cv2.imread("files/samples/girl.png")

    # Preprocess for BlazePose Detector
    img1, img2, scale, pad = resize_pad(orig_image)
    img2 = torch.from_numpy(img2).permute((2, 0, 1)).unsqueeze(0)
    img2 = img2.float() / 255.0
    module_name = build_module_name(framework="pt", model="blazepose", task="palm")
    compiled_model = forge.compile(palm_detector, sample_inputs=[img2], module_name=module_name)


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_blaze_hand_pytorch(record_forge_property):
    # Load BlazePalm Detector
    hand_regressor = BlazeHandLandmark()
    hand_regressor.load_weights("mediapipepytorch/blazehand_landmark.pth")

    sample_tensor = [torch.rand(1, 3, 256, 256)]
    module_name = build_module_name(framework="pt", model="blazepose", task="hand")
    compiled_model = forge.compile(hand_regressor, sample_inputs=sample_tensor, module_name=module_name)
