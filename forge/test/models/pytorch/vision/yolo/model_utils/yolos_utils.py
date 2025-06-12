# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def load_model(variant):
    model = AutoModelForObjectDetection.from_pretrained(variant)
    model.eval()
    return model


def load_input(variant):
    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]
    image_processor = AutoImageProcessor.from_pretrained(variant)
    inputs = image_processor(images=image, return_tensors="pt")
    return [inputs["pixel_values"]]
