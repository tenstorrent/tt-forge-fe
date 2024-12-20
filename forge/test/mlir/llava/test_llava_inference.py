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

    # Extract tensors from inputs
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    # check whether all of the inputs are tensors
    if not all(isinstance(input, torch.Tensor) for input in [input_ids, attention_mask, pixel_values]):
        raise ValueError("All inputs should be tensors")

    # Extract the vision encoder
    vision_encoder = model.vision_tower

    tokens = vision_encoder(pixel_values).last_hidden_state

    class TraceableVisionEncoder(torch.nn.Module):
        def __init__(self, vision_encoder):
            super(TraceableVisionEncoder, self).__init__()
            self.vision_encoder = vision_encoder

        def forward(self, pixel_values):
            # Extract the last hidden state from the vision encoder
            output = self.vision_encoder(pixel_values)

            # Assuming the output is a complex object, we need to extract the tensor we want to trace
            # Here, I assume it's the 'last_hidden_state' from the output, but this depends on the model.
            return output.last_hidden_state

    # Wrap the vision encoder to make it traceable
    vision_encoder = model.vision_tower
    vision_encoder = TraceableVisionEncoder(vision_encoder)

    # Trace the wrapped vision encoder
    traced_vision_encoder = torch.jit.trace(vision_encoder, (pixel_values,))

    # Extract the multi-modal projector
    projector = model.multi_modal_projector

    projected_tokens = projector(tokens)

    traced_projector = torch.jit.trace(projector, (tokens,))

    # Extract the text decoder
    text_decoder = model.language_model

    # Define a wrapper module to extract logits for tracing
    class TextDecoderWrapper(torch.nn.Module):
        def __init__(self, text_decoder):
            super().__init__()
            self.text_decoder = text_decoder

        def forward(self, input_ids, attention_mask, projected_tokens):
            # Use the projected tokens as input to the text decoder
            # Assuming the decoder accepts both input_ids, attention_mask, and projected tokens
            outputs = self.text_decoder(input_ids, attention_mask, encoder_hidden_states=projected_tokens)
            # Return only the logits for tracing
            return outputs.logits

    # Wrap the text decoder
    wrapped_decoder = TextDecoderWrapper(text_decoder)

    # Trace the wrapped model
    traced_text_model = torch.jit.trace(wrapped_decoder, (input_ids, attention_mask, projected_tokens))


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

    output = model(input_ids, attention_mask, pixel_values)
    decoded_output = processor.decode(output[0], skip_special_tokens=True)

    print(decoded_output)
