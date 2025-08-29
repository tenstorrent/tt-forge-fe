# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

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
from third_party.tt_forge_models.mobilenetv2.pytorch import ModelLoader, ModelVariant


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.mobilenet_v2.layer[0].reduce_1x1

    def forward(self, features):
        op = self.model(features)
        return op


variants = [
    pytest.param(ModelVariant.MOBILENET_V2_035_96_HF),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mobilenetv2_hf(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MOBILENETV2,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model.config.return_dict = False

    framework_model = Wrapper(framework_model)

    logger.info("framework_model={}", framework_model)

    inputs = [torch.load("features.pt")]

    features = inputs[0]

    logger.info("framework_model={}", framework_model)
    logger.info("features.shape={}", features.shape)
    logger.info("features.dtype={}", features.dtype)
    logger.info("features={}", features)

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
