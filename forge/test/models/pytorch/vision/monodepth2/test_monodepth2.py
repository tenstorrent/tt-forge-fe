# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.pytorch.vision.monodepth2.model_utils.utils import (
    download_model,
    load_input,
    load_model,
)

variants = [
    "mono_640x192",
    "stereo_640x192",
    "mono+stereo_640x192",
    "mono_no_pt_640x192",
    "stereo_no_pt_640x192",
    "mono+stereo_no_pt_640x192",
    "mono_1024x320",
    "stereo_1024x320",
    "mono+stereo_1024x320",
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_monodepth2(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MONODEPTH2,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.DEPTH_PREDICTION,
    )

    # prepare model and input
    download_model(variant)
    framework_model, height, width = load_model(variant)
    framework_model.to(torch.bfloat16)
    input_tensor = load_input(height, width)

    inputs = [input_tensor.to(torch.bfloat16)]
    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    pcc = 0.99

    if variant in ["stereo_640x192", "mono_no_pt_640x192", "stereo_no_pt_640x192"]:
        pcc = 0.98

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
