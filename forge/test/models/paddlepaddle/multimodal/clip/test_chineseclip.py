# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import requests
from PIL import Image

import pytest

import paddle
from paddlenlp.transformers import (
    ChineseCLIPProcessor,
    ChineseCLIPTokenizer,
    ChineseCLIPModel,
    ChineseCLIPTextModel,
    ChineseCLIPVisionModel,
)

from forge.tvm_calls.forge_utils import paddle_trace
import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task

variants = ["OFA-Sys/chinese-clip-vit-base-patch16"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_chineseclip_text(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="chineseclip_text",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.TEXT_ENCODING,
    )

    # Load Model and Tokenizer
    model = ChineseCLIPTextModel.from_pretrained(variant)
    model.eval()
    tokenizer = ChineseCLIPTokenizer.from_pretrained(variant)

    # Load sample
    inputs = tokenizer("一只猫的照片", padding=True, return_tensors="pd")
    inputs = [inputs["input_ids"]]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_chineseclip_vision(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="chineseclip_vision",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_ENCODING,
    )

    # Load Model and Tokenizer
    model = ChineseCLIPVisionModel.from_pretrained(variant)
    model.eval()
    processor = ChineseCLIPProcessor.from_pretrained(variant)

    # Load sample
    image = Image.open(
        requests.get("https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg", stream=True).raw
    )

    inputs = processor(images=image, return_tensors="pd")
    inputs = [inputs["pixel_values"]]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_chineseclip(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="chineseclip",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_TEXT_PAIRING,
    )

    # Load Model and Tokenizer
    model = ChineseCLIPModel.from_pretrained(variant)
    model.eval()
    processor = ChineseCLIPProcessor.from_pretrained(variant)

    # Load sample
    text = ["椅子", "玫瑰", "小火龙", "皮卡丘"]
    image = Image.open(
        requests.get("https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg", stream=True).raw
    )
    inputs = processor(images=image, text=text, return_tensors="pd", padding=True)
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

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
