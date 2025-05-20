# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import requests

# From: https://huggingface.co/alibaba-damo/mgp-str-base
from PIL import Image
from transformers import MgpstrForSceneTextRecognition, MgpstrProcessor


def load_model(variant):
    model = MgpstrForSceneTextRecognition.from_pretrained(variant)
    model.eval()
    return model


def load_input(variant):
    url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    processor = MgpstrProcessor.from_pretrained(variant)
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    return [inputs["pixel_values"]]
