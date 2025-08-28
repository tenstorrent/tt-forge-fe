# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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

from third_party.tt_forge_models.vgg19_unet.pytorch import ModelLoader  # isort:skip


@pytest.mark.nightly
@pytest.mark.xfail
def test_vgg19_unet():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG19UNET,
        variant="default",
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.GITHUB,
    )
    pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2888")

    # Load model and input
    loader = ModelLoader()
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_sample = loader.load_inputs(dtype_override=torch.bfloat16)

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_sample],
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify([input_sample], framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
def test_vgg19_unet_brain_tumor_segmentation():
    # Record Forge Property
    record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG19UNET,
        variant="default",
        task=Task.BRAIN_TUMOR_SEGMENTATION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Test is currently not executable due to model code dependency.")
