# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.pytorch.vision.mnist.model_utils.utils import load_input, load_model


@pytest.mark.nightly
def test_mnist():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="mnist",
        source=Source.GITHUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and input
    framework_model = load_model().to(torch.bfloat16)
    inputs = load_input()
    inputs = [inputs.to(torch.bfloat16)]

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
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)),
    )
