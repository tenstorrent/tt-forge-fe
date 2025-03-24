# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
import requests
import os
import torch
from PIL import Image
from torchvision import transforms


def load_inputs(img):
    input_tensors = []
    input_shape = [1, 3, 224, 224]
    input_height = input_shape[2]
    input_width = input_shape[3]
    transform = transforms.Compose(
        [
            transforms.Resize((input_height, input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply the transformation
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    input_tensors.append(img_tensor)

    return input_tensors
