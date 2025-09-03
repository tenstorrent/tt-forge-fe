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

# from test.models.models_utils import print_cls_results
from test.utils import download_model
from loguru import logger
import torch.nn as nn 

varaints = [
        "mixer_b16_224.goog_in21k",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_mlp_mixer_timm_onnx(variant, forge_tmp_path):
    
    class Wrapper(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.model = model.norm

        def forward(self,x):

            op = self.model(x)
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
    
    x = torch.load('norm_ip.pt').contiguous()
    logger.info("x={}",x)
    logger.info("x.shape={}",x.shape)
    logger.info("x.dtype={}",x.dtype)
    
    inputs = [x]

    logger.info("inputs={}",inputs)

    # Export model to ONNX
    onnx_path = f"forge/test/models/onnx/vision/mlp_mixer_debug/s1.onnx"
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


