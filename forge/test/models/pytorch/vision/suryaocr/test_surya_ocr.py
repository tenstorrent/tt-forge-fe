# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import io
import urllib.request
from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.suryaocr.model_utils.model_utils import (
    SuryaOCRWrapper,
    freeze_all,
    save_outputs,
)


def test_surya_ocr():

    # Hardcode image(s) from URL
    IMAGE_URL = "https://raw.githubusercontent.com/VikParuchuri/surya/master/static/images/excerpt_text.png"
    with urllib.request.urlopen(IMAGE_URL) as resp:
        img_bytes = resp.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    images: List[Image.Image] = [image]
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    image_tensor = torch.stack([image_tensor])

    # Load model
    framework_model = SuryaOCRWrapper()
    freeze_all(framework_model, warmup_input=image_tensor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=image_tensor, module_name="surya_ocr")

    # Model Verification
    _, co_out = verify(image_tensor, framework_model, compiled_model)

    save_outputs(co_out, images)
