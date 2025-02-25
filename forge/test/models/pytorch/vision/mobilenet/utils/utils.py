# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import urllib

import requests
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from test.models.pytorch.vision.mobilenet.utils.mobilenet_v1 import MobileNetV1
from test.utils import download_model


def load_mobilenet_model(model_name):

    # Create model
    if model_name == "mobilenet_v1":
        model = MobileNetV1(9)
    else:
        model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", model_name, pretrained=True)

    model.eval()

    # Load data sample
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)

    # Preprocessing
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return model, [input_batch]


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
    os.remove("dog.jpg")


def load_model():
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    return model


def load_input():
    weights = models.MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    return [batch_t]
