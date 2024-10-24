# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from PIL import Image
import requests
from torchvision import transforms
import os
import pytest
from test.model_demos.models.retinanet.model_implementation import Model
import zipfile
import shutil
import forge


def img_preprocess():

    url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    new_size = (640, 480)
    pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img


variants = [
    "retinanet_rn18fpn",
    "retinanet_rn34fpn",
    "retinanet_rn50fpn",
    "retinanet_rn101fpn",
    "retinanet_rn152fpn",
]


@pytest.mark.parametrize("variant", variants)
def test_retinanet(variant, test_device):

    # Set PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Prepare model
    url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant}.zip"
    local_zip_path = f"{variant}.zip"

    response = requests.get(url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Find the path of the .pth file
    extracted_path = f"{variant}"
    checkpoint_path = ""
    for root, _, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".pth"):
                checkpoint_path = os.path.join(root, file)

    model = Model.load(checkpoint_path)
    model.eval()

    # Prepare input
    input_batch = img_preprocess()
    inputs = [input_batch]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{variant}")

    # Delete the extracted folder and the zip file
    shutil.rmtree(extracted_path)
    os.remove(local_zip_path)
