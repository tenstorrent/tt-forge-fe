# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
import transformers
import forge
from test.mlir.llava.utils.utils import load_llava_model


@pytest.mark.nightly
# @pytest.mark.xfail(reason="AttributeError: module 'jaxlib.xla_extension' has no attribute 'DeviceArray'")
@pytest.mark.parametrize("model_path", ["llava-hf/llava-1.5-7b-hf"])
def test_llava_compile(model_path):
    model, processor = load_llava_model(model_path)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown on the image? Come up with a story of how image was taken."},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image = torch.randint(0, 256, (3, 224, 224))  # Generate random image
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # Ensure inputs is a dictionary of tensors
    if isinstance(inputs, transformers.feature_extraction_utils.BatchFeature):
        inputs = {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}

    # Extract input_ids tensor from inputs
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("input_ids not found in inputs")

    # Extract other necessary tensors
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    # Ensure all inputs are tensors
    model_inputs = [input_ids, attention_mask, pixel_values]

    class MyLLavaModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, pixel_values):
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=200,
                do_sample=False,
            )

    # Wrap the model in a torch.nn.Module
    model = MyLLavaModel(model)

    # Compile the model
    compiled_model = forge.compile(model, model_inputs)


@pytest.mark.nightly
@pytest.mark.skip(reason="No need to run in CI, this is PoC that should be mapped to work on device.")
@pytest.mark.parametrize("model_path", ["llava-hf/llava-1.5-7b-hf"])
def test_llava_inference(model_path):
    model, processor = load_llava_model(model_path)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown on the image? Come up with a story of how image was taken."},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image = torch.randint(0, 256, (3, 224, 224))

    inputs = processor(images=image, text=prompt, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    decoded_output = processor.decode(output[0], skip_special_tokens=True)

    print(decoded_output)
