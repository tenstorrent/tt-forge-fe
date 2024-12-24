# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
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

    inputs = [pixel_values]

    print("Input IDs Shape:", input_ids.shape)
    print("Attention Mask Shape:", attention_mask.shape)
    print("Pixel Values Shape:", pixel_values.shape)

    # check whether all of the inputs are tensors
    if not all(isinstance(input, torch.Tensor) for input in [input_ids, attention_mask, pixel_values]):
        raise ValueError("All inputs should be tensors")

    class LLavaVisionEncoder(torch.nn.Module):
        def __init__(self, vision_tower):
            super().__init__()
            self.vision_encoder = vision_tower

        def forward(self, pixel_values):
            vision_features = self.vision_encoder(pixel_values).last_hidden_state
            return vision_features

    # Wrap the model in our torch.nn.Module
    framework_model = LLavaVisionEncoder(model.vision_tower)

    # Compile the model
    compiled_model = forge.compile(framework_model, inputs)
