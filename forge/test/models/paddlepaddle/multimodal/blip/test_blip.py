# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import requests
from PIL import Image

import pytest

import paddle
from paddlenlp.transformers import BlipProcessor, BlipTextModel, BlipVisionModel, BlipModel

from forge.tvm_calls.forge_utils import paddle_trace
import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["Salesforce/blip-image-captioning-base"]


@pytest.mark.parametrize("variant", variants)
def test_blip_text(variant):
    model = BlipTextModel.from_pretrained(variant)
    processor = BlipProcessor.from_pretrained(variant)

    text = "a photo of cats in bed"
    inputs = processor(text=text, return_tensors="pd", padding=True)
    inputs = [inputs["input_ids"]]

    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_blip_vision(variant):
    model = BlipVisionModel.from_pretrained(variant)
    processor = BlipProcessor.from_pretrained(variant)

    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    inputs = processor(images=image, return_tensors="pd", padding=True)
    inputs = [inputs["pixel_values"]]

    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_blip(variant, forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="blip",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_ENCODING,
    )

    # Record Forge Property
    forge_property_recorder.record_model_name(module_name)
    forge_property_recorder.record_group("generality")

    # Load Model and Tokenizer
    model = BlipModel.from_pretrained(variant)
    processor = BlipProcessor.from_pretrained(variant)

    class BlipWrapper(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, pixel_values, attention_mask):
            output = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            return output.text_embeds, output.image_embeds

    model = BlipWrapper(model)

    # Prepare inputs
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    text = [
        "cats sleeping",
        "snowy weather",
    ]
    inputs = processor(images=image, text=text, return_tensors="pd", padding=True)

    inputs = [inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]]

    # Test framework model
    outputs = model(*inputs)

    image_embed = outputs[1]
    text_embeds = outputs[0]

    image_embed = paddle.nn.functional.normalize(image_embed, axis=-1)
    text_embeds = paddle.nn.functional.normalize(text_embeds, axis=-1)

    similarities = paddle.matmul(text_embeds, image_embed.T)
    similarities = similarities.squeeze().numpy()

    for t, sim in zip(text, similarities):
        print(f"{t}: similarity = {sim:.4f}")

    # Compile model
    compiled_model = forge.compile(
        model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Verify
    verify(inputs, model, compiled_model, forge_property_handler=forge_property_recorder)
