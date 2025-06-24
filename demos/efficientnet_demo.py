# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Efficientnet Demo Script

import forge
import torch
from PIL import Image
import timm
import urllib
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def run_efficientnet_pytorch(variant="efficientnet_b0", batch_size=1):

    # Load model
    framework_model = timm.create_model(variant, pretrained=True)
    framework_model.eval()

    # Load and pre-process image
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    input_tensor = transform(img).unsqueeze(0)
    input = [input_tensor] * batch_size
    batch_input = torch.cat(input, dim=0)

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=[batch_input])

    # Run inference on Tenstorrent device
    output = compiled_model(batch_input)

    # Post-process output
    probabilities = torch.nn.functional.softmax(output[0], dim=1)
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    for b in range(batch_size):
        print(f"\nTop-5 predictions for batch {b}:")
        top5_prob, top5_catid = torch.topk(probabilities[b], 5)
        for i in range(5):
            print(f"{categories[top5_catid[i]]}: {top5_prob[i].item():.4f}")


if __name__ == "__main__":
    run_efficientnet_pytorch()
