# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import skimage
import torch
import torchvision
import torchxrayvision as xrv
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from torchvision.transforms import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    Normalize,
    PILToTensor,
    Resize,
)


def get_input_img():
    try:
        input_image = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(str(input_image)).convert("RGB")

        transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Preprocessing
        img_tensor = transform(img).unsqueeze(0)

        # Make the tensor contiguous.
        # Current limitation of compiler/runtime is that it does not support non-contiguous tensors properly.
        img_tensor = img_tensor.contiguous()
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)
    print(img_tensor.shape)
    return img_tensor


def get_input_img_hf_xray():
    try:
        img_path = get_file(
            "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
        )
        img = skimage.io.imread(str(img_path))
        img = xrv.datasets.normalize(img, 255)
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        # Add color channel
        img = img[None, :, :]
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 1, 224, 224)

    return img_tensor
