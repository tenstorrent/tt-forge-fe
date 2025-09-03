# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import timm
import torch
import onnx
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model
from loguru import logger
import torch.nn as nn 

def test_mlp_mixer_timm_onnx_cases(forge_tmp_path):
    
    variant = "mixer_b16_224.goog_in21k"
    
    class Wrapper(nn.Module):
        def __init__(self,model):
            super().__init__()
            
            self.stem = model.stem
            self.b0 = model.blocks[0]
            self.b1 = model.blocks[1]
            self.b2 = model.blocks[2]
            self.b3 = model.blocks[3]
            self.b4 = model.blocks[4]
            self.b5 = model.blocks[5]
            
            self.b6 = model.blocks[6]
            self.b7 = model.blocks[7]
            self.b8 = model.blocks[8]
            self.b9 = model.blocks[9]
            self.b10 = model.blocks[10]
            self.b11 = model.blocks[11]
            self.norm = model.norm
            

        def forward(self,x):

            # x = self.stem(x)
            # x = self.b0(x)
            # x = self.b1(x)
            # x = self.b2(x)
            # x = self.b3(x)
            # x = self.b4(x)
            # x = self.b5(x)
            # x = self.b6(x)
            # x = self.b7(x)
            # x = self.b8(x)
            # x = self.b9(x)
            x = self.b10(x)
            x = self.b11(x)
            op = self.norm(x)
            return op
    
    
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MLPMIXER,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    load_pretrained_weights = True
    if variant in ["mixer_s32_224", "mixer_s16_224", "mixer_b32_224", "mixer_l32_224"]:
        load_pretrained_weights = False

    framework_model = download_model(timm.create_model, variant, pretrained=load_pretrained_weights)
    
    framework_model = Wrapper(framework_model)
    
    logger.info("framework_model={}",framework_model)
    inputs = [torch.load('b10_ip.pt').contiguous()]
    

    # Export model to ONNX
    onnx_path = f"forge/test/models/onnx/vision/mlp_mixer_debug/b10_to_norm.onnx"
    torch.onnx.export(
        framework_model,
        (inputs[0],),
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=True,
    )

    # Load and verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify model
    fw_out, co_out = verify(inputs, framework_model, compiled_model)


