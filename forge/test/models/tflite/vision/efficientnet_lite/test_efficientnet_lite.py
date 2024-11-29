# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from forge.config import _get_global_compiler_config
import forge


@pytest.mark.nightly
@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite0_1x1(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite0-fp32.tflite"
    sample_tensor = [torch.rand(1, 224, 224, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.nightly
@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite4_1x1(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite4-fp32.tflite"
    sample_tensor = [torch.rand(1, 320, 320, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.nightly
@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite0(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite0-fp32.tflite"
    sample_tensor = [torch.rand(1, 224, 224, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
def test_efficientnet_lite1(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite1-fp32.tflite"
    sample_tensor = [torch.rand(1, 240, 240, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
def test_efficientnet_lite2(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite2-fp32.tflite"
    sample_tensor = [torch.rand(1, 260, 260, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
def test_efficientnet_lite3(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite3-fp32.tflite"
    sample_tensor = [torch.rand(1, 280, 280, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)


@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
def test_efficientnet_lite4(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/efficientnet-lite4-fp32.tflite"
    sample_tensor = [torch.rand(1, 320, 320, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)
