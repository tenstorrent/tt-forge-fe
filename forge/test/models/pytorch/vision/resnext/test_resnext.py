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
from forge.verify.verify import verify
from third_party.tt_forge_models.resnext.pytorch import ModelLoader, ModelVariant

variants = [ModelVariant.RESNEXT50_32X4D, ModelVariant.RESNEXT101_32X8D, ModelVariant.RESNEXT101_32X8D_WSL]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_resnext_torchhub_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNEXT,
        source=Source.TORCH_HUB,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    loader.print_cls_results(co_out)


variants = [
    ModelVariant.RESNEXT14_32X4D_OSMR,
    ModelVariant.RESNEXT26_32X4D_OSMR,
    ModelVariant.RESNEXT50_32X4D_OSMR,
    ModelVariant.RESNEXT101_64X4D_OSMR,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_resnext_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNEXT,
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.print_cls_results(co_out)
