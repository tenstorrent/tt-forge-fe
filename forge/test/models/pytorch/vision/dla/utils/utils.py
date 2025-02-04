# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import urllib

import requests
import torch
import torchvision.transforms as transforms
from PIL import Image

from test.models.pytorch.vision.dla.utils import dla_model


def load_dla_model(variant):

    func = getattr(dla_model, variant)

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

    framework_model = func(pretrained="imagenet")
    framework_model.eval()

    inputs = [img_tensor]

    return framework_model, inputs


url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    urllib.request.urlretrieve(url, "imagenet_classes.txt")

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())

    # Cleanup
    os.remove("imagenet_classes.txt")
