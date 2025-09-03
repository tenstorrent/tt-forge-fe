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

varaints = [
        "mixer_b16_224.goog_in21k",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_mlp_mixer_timm_onnx(variant, forge_tmp_path):
    
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
            
            torch.save(x,'s_ip.pt')
            x = self.stem(x)
            torch.save(x,'b0_ip.pt')
            x = self.b0(x)
            torch.save(x,'b1_ip.pt')
            x = self.b1(x)
            torch.save(x,'b2_ip.pt')
            x = self.b2(x)
            torch.save(x,'b3_ip.pt')
            x = self.b3(x)
            torch.save(x,'b4_ip.pt')
            x = self.b4(x)
            torch.save(x,'b5_ip.pt')
            x = self.b5(x)
            torch.save(x,'b6_ip.pt')
            x = self.b6(x)
            torch.save(x,'b7_ip.pt')
            x = self.b7(x)
            torch.save(x,'b8_ip.pt')
            x = self.b8(x)
            torch.save(x,'b9_ip.pt')
            x = self.b9(x)
            torch.save(x,'b10_ip.pt')
            x = self.b10(x)
            torch.save(x,'b11_ip.pt')
            x = self.b11(x)
            torch.save(x,'norm_ip.pt')
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
    
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)

    try:
        if variant in [
            "mixer_b16_224_in21k",
            "mixer_b16_224_miil_in21k",
            "mixer_l16_224_in21k",
            "mixer_b16_224.goog_in21k",
        ]:
            input_image = get_file(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
            use_1k_labels = False
        else:
            input_image = get_file(
                "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
            )
            use_1k_labels = True
        image = Image.open(str(input_image)).convert("RGB")
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image = torch.rand(1, 3, 256, 256)
    pixel_values = transform(image).unsqueeze(0)

    inputs = [pixel_values]
    
    op = framework_model(*inputs)
    
    print("op=",op)

    # # Export model to ONNX
    # onnx_path = f"forge/test/models/onnx/vision/mlp_mixer_debug/models/s5.onnx"
    # torch.onnx.export(
    #     framework_model,
    #     (inputs[0],),
    #     onnx_path,
    #     opset_version=17,
    #     input_names=["input"],
    #     output_names=["output"],
    #     keep_initializers_as_inputs=True,
    # )

    # # Load and verify ONNX model
    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    # framework_model = forge.OnnxModule(module_name, onnx_model)

    # # Forge compile ONNX model
    # compiled_model = forge.compile(
    #     onnx_model,
    #     sample_inputs=inputs,
    #     module_name=module_name,
    # )

    # # Verify model
    # fw_out, co_out = verify(inputs, framework_model, compiled_model)


