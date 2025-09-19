# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.unet.pytorch import ModelLoader, ModelVariant

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.unet.model_utils.model import UNET


@pytest.mark.xfail
@pytest.mark.nightly
def test_unet_osmr_cityscape_pytorch():
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNET,
        variant="cityscape",
        source=Source.OSMR,
        task=Task.IMAGE_SEGMENTATION,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    loader = ModelLoader(variant=ModelVariant.OSMR_CITYSCAPES)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = [loader.load_inputs(dtype_override=torch.bfloat16)]

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
def test_unet_qubvel_pytorch():
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNET,
        variant="qubvel",
        source=Source.TORCH_HUB,
        task=Task.IMAGE_SEGMENTATION,
    )
    pytest.xfail("https://github.com/tenstorrent/tt-forge-fe/issues/2940")

    loader = ModelLoader(variant=ModelVariant.SMP_UNET_RESNET101)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = [loader.load_inputs(dtype_override=torch.bfloat16)]

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
def test_unet_torchhub_pytorch():
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNET,
        source=Source.TORCH_HUB,
        task=Task.IMAGE_SEGMENTATION,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )
    pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2956")

    loader = ModelLoader(variant=ModelVariant.TORCHHUB_BRAIN_UNET)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = [loader.load_inputs(dtype_override=torch.bfloat16)]

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify(inputs, framework_model, compiled_model)


# Reference: https://github.com/arief25ramadhan/carvana-unet-segmentation
@pytest.mark.nightly
def test_unet_carvana():
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNETCARVANA,
        source=Source.GITHUB,
        task=Task.IMAGE_SEGMENTATION,
    )

    framework_model = UNET(in_channels=3, out_channels=1).to(torch.bfloat16)
    framework_model.eval()
    inputs = [torch.rand((1, 3, 224, 224)).to(torch.bfloat16)]

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify(inputs, framework_model, compiled_model)
