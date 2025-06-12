# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from transformers import BeitForImageClassification, BeitImageProcessor


def load_model(variant):
    model = BeitForImageClassification.from_pretrained(variant)
    model.eval()
    return model


def load_input(variant):
    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]
    processor = BeitImageProcessor.from_pretrained(variant)
    inputs = processor(images=image, return_tensors="pt")
    return [inputs["pixel_values"]]
