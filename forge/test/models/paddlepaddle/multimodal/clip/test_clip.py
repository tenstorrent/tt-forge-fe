# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import requests
from PIL import Image

import pytest

import paddle
from paddlenlp.transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel

from forge.tvm_calls.forge_utils import paddle_trace
import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task

variants = ["openai/clip-vit-base-patch16"]


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_clip_text(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="clip_text",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.TEXT_ENCODING,
    )
    forge_property_recorder.record_group("generality")

    # Load model and processor
    model = CLIPTextModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    # Prepare inputs
    text = "a photo of cats in bed"
    inputs = processor(text=text, return_tensors="pd", padding=True)
    inputs = [inputs["input_ids"]]

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_clip_vision(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="clip_vision",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_ENCODING,
    )
    forge_property_recorder.record_group("generality")

    # Load model and processor
    model = CLIPVisionModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    # Prepare inputs
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    inputs = processor(images=image, return_tensors="pd")
    inputs = [inputs["pixel_values"]]

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail()
def test_clip(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="clip",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_TEXT_PAIRING,
    )
    forge_property_recorder.record_group("generality")

    # Load model and processor
    model = CLIPModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    # Prepare inputs
    text = [
        "a photo of cats in bed",
        "a photo of dog in snow",
    ]
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    inputs = processor(images=image, text=text, return_tensors="pd")
    inputs = [inputs["input_ids"], inputs["pixel_values"]]

    # Test framework model
    outputs = model(*inputs)

    image_embed = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embed = paddle.nn.functional.normalize(image_embed, axis=-1)
    text_embeds = paddle.nn.functional.normalize(text_embeds, axis=-1)

    similarities = paddle.matmul(text_embeds, image_embed.T)
    similarities = similarities.squeeze().numpy()

    for t, sim in zip(text, similarities):
        print(f"{t}: similarity = {sim:.4f}")

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
