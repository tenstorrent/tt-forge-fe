# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import timm
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

from test.utils import download_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.stem = model.norm

    def forward(self, x):
        x = self.norm(x)
        return x


varaints = [
    pytest.param(
        "mixer_b16_224",
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_mlp_mixer_timm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MLPMIXER,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    load_pretrained_weights = True
    if variant in ["mixer_s32_224", "mixer_s16_224", "mixer_b32_224", "mixer_l32_224"]:
        load_pretrained_weights = False

    framework_model = download_model(timm.create_model, variant, pretrained=load_pretrained_weights).to(torch.bfloat16)
    framework_model = Wrapper(framework_model).to(torch.bfloat16)

    inputs = [torch.load("x.pt")]

    logger.info("framework_model={}", framework_model)

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
    fw_out, co_out = verify(inputs, framework_model, compiled_model)
