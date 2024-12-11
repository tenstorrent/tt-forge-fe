# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from io import BytesIO
import requests
import pytest
import torch
from PIL import Image
import PIL.Image as pil
from torchvision import transforms
import forge
from test.models.pytorch.vision.monodepth2.utils.utils import download_model_if_doesnt_exist
from test.models.pytorch.vision.monodepth2.utils.resnet_encoder import ResnetEncoder
from test.models.pytorch.vision.monodepth2.utils.depth_decoder import DepthDecoder


variants = [
    "mono_640x192",
    "stereo_640x192",
    "mono+stereo_640x192",
    "mono_no_pt_640x192",
    "stereo_no_pt_640x192",
    "mono+stereo_no_pt_640x192",
    "mono_1024x320",
    "stereo_1024x320",
    "mono+stereo_1024x320",
]


@pytest.mark.parametrize("variant", variants)
def test_monodepth2(variant):

    # PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # prepare model
    download_model_if_doesnt_exist(variant)
    encoder_path = os.path.join("models", variant, "encoder.pth")
    depth_decoder_path = os.path.join("models", variant, "depth.pth")

    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
    depth_decoder.load_state_dict(loaded_dict)

    class MonoDepth2(torch.nn.Module):
        def __init__(self, encoder, depth_decoder):
            super().__init__()
            self.encoder = encoder
            self.depth_decoder = depth_decoder

        def forward(self, input):
            features = encoder(input)
            outputs = depth_decoder(features)
            return outputs[("disp", 0)]

    model = MonoDepth2(encoder, depth_decoder)
    model.eval()

    # prepare input
    image_url = "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/assets/test_image.jpg"
    response = requests.get(image_url)
    input_image = Image.open(BytesIO(response.content)).convert("RGB")
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    # Forge inference
    compiled_model = forge.compile(
        model, sample_inputs=[input_tensor], module_name=f"pt_{variant.replace('x', '_').replace('+', '_')}"
    )
