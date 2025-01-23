# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torchvision.models.resnet import resnet50

import forge
from forge.verify.verify import verify


@pytest.mark.push
def test_resnet_inference():
    # Compiler configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    # Load ResNet50 model
    framework_model = resnet50()
    framework_model.eval()

    input_image = torch.rand(1, 3, 224, 224)
    inputs = [input_image]

    # Compile the model
    compiled_model = forge.compile(framework_model, input_image)

    verify(inputs, framework_model, compiled_model)
