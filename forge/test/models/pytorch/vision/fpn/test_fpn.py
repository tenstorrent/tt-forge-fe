# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.vision.fpn.model_utils.model import FPNWrapper


@pytest.mark.nightly
def test_fpn_pytorch():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model="fpn", source=Source.TORCHVISION, task=Task.IMAGE_CLASSIFICATION
    )

    # Load FPN model
    framework_model = FPNWrapper().to(torch.bfloat16)

    feat0 = torch.rand(1, 256, 64, 64).to(torch.bfloat16)
    feat1 = torch.rand(1, 512, 16, 16).to(torch.bfloat16)
    feat2 = torch.rand(1, 2048, 8, 8).to(torch.bfloat16)

    inputs = [feat0, feat1, feat2]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
