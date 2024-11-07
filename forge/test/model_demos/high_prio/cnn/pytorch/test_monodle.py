# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import requests
import torchvision.transforms as transforms
from PIL import Image
from test.model_demos.models.monodle import CenterNet3D
import os


@pytest.mark.nightly
def test_monodle_pytorch(test_device):
    # PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    model_name = "monodle_pytorch"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    pytorch_model = CenterNet3D(backbone="dla34")
    pytorch_model.eval()
    compiled_model = forge.compile(pytorch_model, sample_inputs=[img_tensor], module_name="pt_monodle")
