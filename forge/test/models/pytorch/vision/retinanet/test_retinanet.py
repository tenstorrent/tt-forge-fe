# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Import the ModelLoader from tt-forge

import pytest
import torch
from third_party.tt_forge_models.retinanet.pytorch.loader import (
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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.models.pytorch.vision.retinanet.model_utils.model_utils import Wrapper

variants = [
    ModelVariant.RETINANET_RN18FPN,
    ModelVariant.RETINANET_RN34FPN,
    ModelVariant.RETINANET_RN50FPN,
    ModelVariant.RETINANET_RN101FPN,
    ModelVariant.RETINANET_RN152FPN,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_retinanet(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RETINANET,
        variant=variant.value,
        source=Source.GITHUB,
        task=Task.OBJECT_DETECTION,
    )

    # Load model and inputs using ModelLoader
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
    verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))
    )


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [ModelVariant.RETINANET_RESNET50_FPN_V2])
def test_retinanet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RETINANET,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )
    pytest.xfail(reason="Fatal Python error: Aborted")

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model = Wrapper(model)
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
    verify(inputs, framework_model, compiled_model)
