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


@pytest.mark.nightly_models_ops
def test_s():
    class Divide0(nn.Module):
        def __init__(self):
            super().__init__()
            self.const = torch.load('const_ip.pt')

        def forward(self, x):
            return torch.div(self.const, x)


    input_tensor = torch.load('divide_input_1.pt')

    inputs = [input_tensor]

    model = Divide0().to(torch.bfloat16)
    
    torch.set_printoptions(precision=6, threshold=10000, linewidth=200, sci_mode=False)
        
    print("inputs=",inputs)

    
    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    
    compiled_model = forge.compile(model, sample_inputs=inputs,compiler_cfg=compiler_cfg,)

    fw_out, co_out = verify(inputs, model, compiled_model)