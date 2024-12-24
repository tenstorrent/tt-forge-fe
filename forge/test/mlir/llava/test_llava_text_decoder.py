# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
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

    class CustomLLavaModel(torch.nn.Module):
        def __init__(self, language_model, image_token_index):
            super().__init__()
            self.text_decoder = language_model
            self.image_token_index = image_token_index

        def forward(self, input_ids, projected_vision_features, attention_mask):
            #  Obtain embeddings from input IDs
            inputs_embeds = self.text_decoder.get_input_embeddings()(input_ids)

            # Inject vision features into input embeddings
            image_token_mask = (input_ids == self.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_token_mask, projected_vision_features)

            # finally we use the language model to generate the text
            return self.text_decoder.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=200, do_sample=False
            )

    vision_features = model.vision_tower(pixel_values).last_hidden_state
    projected_vision_features = model.multi_modal_projector(vision_features)
    inputs = [input_ids, projected_vision_features, attention_mask]

    # Wrap the model in our torch.nn.Module
    framework_model = CustomLLavaModel(model.language_model, model.config.image_token_index)

    traced_model = torch.jit.trace(framework_model, (input_ids, projected_vision_features, attention_mask))

    # Compile the model
    # compiled_model = forge.compile(framework_model, inputs)
