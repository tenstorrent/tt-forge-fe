# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

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

from test.models.pytorch.vision.mobilenet.model_utils.utils import (
    load_mobilenet_model,
    post_processing,
)
from test.models.pytorch.vision.vision_utils.utils import load_timm_model_and_input
from test.utils import download_model
from loguru import logger

import torch.nn as nn 

variants = ["mobilenetv1_100.ra4_e3600_r224_in1k"]


@pytest.mark.nightly
# @pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_s1(variant):
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()


            self.b4_1_bn2 = model.blocks[4][1].bn2


        def forward(self, x):

            x = self.b4_1_bn2(x)

            return x

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MOBILENETV1,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and inputs
    model, _ = load_timm_model_and_input(variant)
    model = model.to(torch.bfloat16)
    inputs = [torch.load('ip_last_bn.pt')]

    framework_model = Wrapper(model)

    logger.info("framework_model={}",framework_model)


    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name="sanity",
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    verify(inputs, framework_model, compiled_model)
