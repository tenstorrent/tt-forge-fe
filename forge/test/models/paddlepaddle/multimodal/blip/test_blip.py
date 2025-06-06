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

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants = ["Salesforce/blip-image-captioning-base"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_blip_text(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.BLIPTEXT,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.TEXT_ENCODING,
    )

    # Load Model and Tokenizer
    model = BlipTextModel.from_pretrained(variant)
    processor = BlipProcessor.from_pretrained(variant)

    # Prepare inputs
    text = "a photo of cats in bed"
    inputs = processor(text=text, return_tensors="pd", padding=True)
    inputs = [inputs["input_ids"]]

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_blip_vision(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.BLIPVISION,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_ENCODING,
    )

    # Load Model and Tokenizer
    model = BlipVisionModel.from_pretrained(variant)
    processor = BlipProcessor.from_pretrained(variant)

    # Prepare inputs
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    inputs = processor(images=image, return_tensors="pd", padding=True)
    inputs = [inputs["pixel_values"]]

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_blip(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.BLIP,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_CAPTIONING,
    )

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
    compiled_model = forge.compile(model, inputs, module_name=module_name)

    # Verify
    verify(inputs, model, compiled_model)
