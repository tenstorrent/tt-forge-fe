# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
import requests
import os
import torch
from PIL import Image
from torchvision import transforms
import timm


def load_inputs(img, model):
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    # Apply the transformation
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    input_tensors.append(img_tensor)

    return input_tensors
