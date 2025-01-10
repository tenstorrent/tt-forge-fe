# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from forge.config import _get_global_compiler_config
import forge


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
def test_mobilenet_ssd_1x1(test_device):
    compiler_cfg = _get_global_compiler_config()
    tflite_path = "third_party/confidential_customer_models/model_2/tflite/ssd_mobilenet_v2.tflite"
    sample_tensor = [torch.rand(1, 256, 256, 3)]
    compiled_model = forge.compile(tflite_path, sample_inputs=sample_tensor)
