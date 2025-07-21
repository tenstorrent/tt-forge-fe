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
def test_mobilenet_v1_timm(variant):
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv_stem = model.conv_stem
            self.bn1 = model.bn1
            self.b0 = model.blocks[0]
            self.b1 = model.blocks[1]
            self.b2 = model.blocks[2]
            self.b3 = model.blocks[3]
            self.b4_0 = model.blocks[4][0]
            self.b4_1_conv = model.blocks[4][1].conv_dw
            self.b4_1_bn1 = model.blocks[4][1].bn1
            self.b4_1_aa = model.blocks[4][1].aa
            self.b4_1_se = model.blocks[4][1].se
            self.b4_1_conv_pw = model.blocks[4][1].conv_pw
            self.b4_1_bn2 = model.blocks[4][1].bn2
            

        def forward(self, x):
            x = self.conv_stem(x)
            x = self.bn1(x)
            x = self.b0(x)
            x = self.b1(x)
            x = self.b2(x)
            x = self.b3(x)
            x = self.b4_0(x)
            x = self.b4_1_conv(x)
            x = self.b4_1_bn1(x)
            x = self.b4_1_aa(x)
            x = self.b4_1_se(x)
            x = self.b4_1_conv_pw(x)
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
    framework_model, inputs = load_timm_model_and_input(variant)
    framework_model = framework_model.to(torch.bfloat16)
    inputs = [inp.to(torch.bfloat16) for inp in inputs]
    
    framework_model = Wrapper(framework_model)
    
    logger.info("framework_model={}",framework_model)
    
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
    verify(inputs, framework_model, compiled_model)
