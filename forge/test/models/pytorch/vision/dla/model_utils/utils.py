# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torchvision.transforms as transforms
from datasets import load_dataset

from test.models.pytorch.vision.dla.model_utils import dla_model


def load_dla_model(variant):

    func = getattr(dla_model, variant)

    # Load data sample
    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]

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
