# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def load_model(variant):
    model = AutoModelForImageSegmentation.from_pretrained(variant, trust_remote_code=True)
    model.eval()
    return model


def load_input():
    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    inputs = transform_image(image).unsqueeze(0)
    return [inputs]
