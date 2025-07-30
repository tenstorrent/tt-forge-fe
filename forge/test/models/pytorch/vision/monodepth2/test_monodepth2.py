# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.monodepth2.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

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

variants = [
    ModelVariant.MONO_640X192,
    ModelVariant.STEREO_640X192,
    ModelVariant.MONO_STEREO_640X192,
    ModelVariant.MONO_NO_PT_640X192,
    ModelVariant.STEREO_NO_PT_640X192,
    ModelVariant.MONO_STEREO_NO_PT_640X192,
    ModelVariant.MONO_1024X320,
    ModelVariant.STEREO_1024X320,
    pytest.param(ModelVariant.MONO_STEREO_1024X320, marks=[pytest.mark.xfail]),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.skip(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2629")
def test_monodepth2(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MONODEPTH2,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.DEPTH_PREDICTION,
    )

    # prepare model and input using ModelLoader
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.to(torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)

    inputs = [input_tensor]
    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override, enable_optimization_passes=True)

    pcc = 0.99

    if variant in ["stereo_640x192", "mono_no_pt_640x192", "stereo_no_pt_640x192"]:
        pcc = 0.98
    elif variant == "stereo_1024x320":
        pcc = 0.95

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post-process and save result
    loader.postprocess_and_save_disparity_map(co_out, f"./disparity_maps/{variant}")
