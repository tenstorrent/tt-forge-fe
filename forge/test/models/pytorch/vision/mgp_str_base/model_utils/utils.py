# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset
from transformers import MgpstrForSceneTextRecognition, MgpstrProcessor


def load_model(variant):
    model = MgpstrForSceneTextRecognition.from_pretrained(variant, return_dict=False)
    model.eval()
    return model


def load_input(variant):
    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]
    processor = MgpstrProcessor.from_pretrained(variant)
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    return [inputs["pixel_values"]], processor
