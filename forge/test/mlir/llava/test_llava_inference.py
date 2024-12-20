# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
import transformers
import forge
from test.mlir.llava.utils.utils import load_llava_model


@pytest.mark.nightly
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

    # Extract tensors from inputs
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    print("Input IDs Shape:", input_ids.shape)
    print("Attention Mask Shape:", attention_mask.shape)
    print("Pixel Values Shape:", pixel_values.shape)

    inputs = [input_ids, attention_mask, pixel_values]

    # check whether all of the inputs are tensors
    if not all(isinstance(input, torch.Tensor) for input in [input_ids, attention_mask, pixel_values]):
        raise ValueError("All inputs should be tensors")

    class CustomLLavaModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_encoder = model.vision_tower
            self.projector = model.multi_modal_projector
            self.language_model = model.language_model
            self.model = model

        def forward(self, input_ids, attention_mask, pixel_values):

            # first we use the vision encoder to extract the tokens from the image
            vision_tokens = self.vision_encoder(pixel_values).last_hidden_state

            # then we use the multi-modal projector to project the tokens into the text embedding space
            projected_tokens = self.projector(vision_tokens)

            # we concatenate the projected tokens with the input_ids
            input_ids = torch.cat([input_ids, projected_tokens], dim=1)

            # finally we use the language model to generate the text
            return self.language_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=200, do_sample=False
            )

    # Wrap the model in our torch.nn.Module
    model = CustomLLavaModel(model)

    #     # # Compile the model
    #     # compiled_model = forge.compile(model, inputs)

    inputs_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }

    # Forward pass through the custom model
    output = model(**inputs_dict)

    # Debugging intermediate outputs
    print("Outputs:", output)  # Check if logits shape is as expecte

    decoded_output = processor.decode(output[0], skip_special_tokens=True)

    # Print the decoded output
    print("Decoded Output:", decoded_output)


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

    # Extract tensors from inputs
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=200,
        do_sample=False,
    )
    decoded_output = processor.decode(output[0], skip_special_tokens=True)

    print(decoded_output)
