# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from test.mlir.llava.utils.utils import load_llava_model

import forge


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

        # Create a wrapper class that will make the model traceable

    class TraceableLLaVa(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_encoder = model.vision_tower
            self.projector = model.multi_modal_projector
            self.language_model = model.language_model
            self.model = model

        def forward(self, input_ids, pixel_values, attention_mask) -> torch.Tensor:
            # Step 1: Obtain embeddings from input IDs
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # Step 2: Extract vision features
            vision_features = self.vision_encoder(pixel_values).last_hidden_state
            projected_vision_features = self.projector(vision_features)

            # Step 3: Inject vision features into input embeddings
            image_token_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_token_mask, projected_vision_features)

            # finally we use the language model to generate the text and return the logits so that we can trace the model
            return self.language_model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).logits

    inputs = [input_ids, pixel_values, attention_mask]

    # Wrap the model in our torch.nn.Module
    # framework_model = TraceableLLaVa(model)

    # Compile the model
    compiled_model = forge.compile(model, inputs)


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
