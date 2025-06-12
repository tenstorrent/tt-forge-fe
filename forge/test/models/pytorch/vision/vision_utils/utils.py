# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from urllib.request import urlopen

import timm
import torch
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import models


def load_timm_model_and_input(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    img = Image.open(
        urlopen("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
    )
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_batch = transforms(img).unsqueeze(0)
    return model, input_batch


def load_vision_model_and_input(variant, task, weight_name):
    if task == "detection":
        weights = getattr(models.detection, weight_name).DEFAULT
        model = getattr(models.detection, variant)(weights=weights)
    else:
        weights = getattr(models, weight_name).DEFAULT
        model = getattr(models, variant)(weights=weights)

    model.eval()

    # Preprocess image
    preprocess = weights.transforms()
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image)).convert("RGB")
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)

    # Make the tensor contiguous.
    # Current limitation of compiler/runtime is that it does not support non-contiguous tensors properly.
    batch_t = batch_t.contiguous()

    return model, [batch_t]
