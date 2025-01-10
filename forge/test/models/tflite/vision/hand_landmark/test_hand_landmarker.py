# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from forge.config import _get_global_compiler_config
import forge


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
def test_hand_landmark_lite_1x1(test_device):
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/hand_landmark_lite.tflite"
    sample_tensor = [torch.rand(1, 224, 224, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
def test_palm_detection_lite_1x1(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/palm_detection_lite.tflite"
    sample_tensor = [torch.rand(1, 192, 192, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)
