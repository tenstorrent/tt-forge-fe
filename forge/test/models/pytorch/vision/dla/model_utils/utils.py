# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from third_party.tt_forge_models.tools.utils import get_file

from test.models.pytorch.vision.dla.model_utils import dla_model


def load_dla_model(variant):

    func = getattr(dla_model, variant)

    # Load data sample
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    image = next(iter(dataset))["image"]

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

    framework_model = func(pretrained=None)
    framework_model.eval()

    inputs = [img_tensor]

    return framework_model, inputs


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    class_file_path = get_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

    with open(class_file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())
