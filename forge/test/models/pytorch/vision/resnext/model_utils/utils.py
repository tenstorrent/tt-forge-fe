# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import requests
import torch
from PIL import Image
from torchvision import transforms

from test.utils import download_model


def get_image_tensor():
    # Preprocessing
    input_image = Image.open(requests.get("https://github.com/pytorch/hub/raw/master/images/dog.jpg", stream=True).raw)
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
    return input_batch


def get_resnext_model_and_input(repo_or_dir, model_name):

    # Create model
    model = download_model(torch.hub.load, repo_or_dir, model_name, pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    return model.to(torch.bfloat16), [input_batch.to(torch.bfloat16)]


url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    r = requests.get(url, allow_redirects=True)
    categories = [s.strip() for s in r.content.decode("utf-8").splitlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())
