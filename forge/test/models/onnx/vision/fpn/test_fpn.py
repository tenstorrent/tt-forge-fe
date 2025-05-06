# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
import os
import pytest

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM repo")
def test_fpn_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX, model="fpn", source=Source.TORCHVISION, task=Task.IMAGE_CLASSIFICATION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load FPN model
    onnx_model_path = "third_party/confidential_customer_models/generated/files/fpn.onnx"
    model_name = f"fpn_onnx"
    onnx_model = onnx.load(onnx_model_path)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    feat0 = torch.rand(1, 10, 64, 64)
    feat1 = torch.rand(1, 20, 16, 16)
    feat2 = torch.rand(1, 30, 8, 8)

    inputs = [feat0, feat1, feat2]

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
