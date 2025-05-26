# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re

import requests
from PIL import Image


def is_url(url):
    regex = r"^(https?)://[^\s/$.?#].[^\s]*$"
    return bool(re.match(regex, url))


def load_inputs(inp_image, text, processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, padding=True, add_generation_prompt=True)
    if is_url(inp_image):
        image = Image.open(requests.get(inp_image, stream=True).raw)
    else:
        if os.path.isfile(inp_image):
            image = Image.open(inp_image)
        else:
            raise ValueError("Input is neither a valid URL nor a valid file path.")

    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return input_ids, attn_mask, pixel_values
