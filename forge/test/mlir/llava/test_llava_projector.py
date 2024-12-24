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

    pixel_values = inputs.get("pixel_values")

    class CustomLLavaModel(torch.nn.Module):
        def __init__(self, multi_modal_projector):
            super().__init__()
            self.projector = multi_modal_projector

        def forward(self, vision_features):
            projected_vision_features = self.projector(vision_features)

            return projected_vision_features

    inputs = [model.vision_tower(pixel_values).last_hidden_state]

    # Wrap the model in our torch.nn.Module
    framework_model = CustomLLavaModel(model.multi_modal_projector)

    # Compile the model
    compiled_model = forge.compile(framework_model, inputs)
