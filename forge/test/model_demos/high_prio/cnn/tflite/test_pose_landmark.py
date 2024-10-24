# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from forge.config import _get_global_compiler_config
import forge


@pytest.mark.skip(reason="Not supported yet")
def test_pose_landmark_lite_1x1(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_lite.tflite"
    input_shape = (1, 256, 256, 3)
    sample_tensor = [torch.rand(1, 256, 256, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.skip(reason="Not supported yet")
def test_pose_landmark_heavy_1x1(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_heavy.tflite"
    input_shape = (1, 256, 256, 3)
    sample_tensor = [torch.rand(1, 256, 256, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.skip(reason="Not supported yet")
def test_pose_landmark_lite(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_lite.tflite"
    sample_tensor = [torch.rand(1, 256, 256, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.skip(reason="Not supported yet")
def test_pose_landmark_heavy(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/pose_landmark_heavy.tflite"
    sample_tensor = [torch.rand(1, 256, 256, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)
