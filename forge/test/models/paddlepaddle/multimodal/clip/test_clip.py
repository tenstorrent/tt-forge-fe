# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import requests
from PIL import Image

import pytest

import paddle
from paddlenlp.transformers import CLIPProcessor, CLIPModel, CLIPVisionModel

from forge.tvm_calls.forge_utils import paddle_trace
import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["openai/clip-vit-base-patch32"])
@pytest.mark.xfail(
    reason="TVMError: relay.concatenate requires all tensors have the same shape on non-concatenating axes"
)
def test_clip(variant):
    model = CLIPModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    text = [
        "a photo of cats in bed",
        "a photo of dog in snow",
    ]
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    inputs = processor(images=image, text=text, return_tensors="pd")

    inputs = [inputs["input_ids"], inputs["pixel_values"]]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model, _ = paddle_trace(model, input_spec)

    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)

    outputs = model(*inputs)

    # Extract embeddings
    image_embed = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Normalize
    image_embed = paddle.nn.functional.normalize(image_embed, axis=-1)
    text_embeds = paddle.nn.functional.normalize(text_embeds, axis=-1)

    # Cosine similarity
    similarities = paddle.matmul(text_embeds, image_embed.T)
    similarities = similarities.squeeze().numpy()

    # Result
    for t, sim in zip(text, similarities):
        print(f"{t}: similarity = {sim:.4f}")


@pytest.mark.parametrize("variant", ["openai/clip-vit-base-patch32"])
def test_clip_vision(variant):
    model = CLIPVisionModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    inputs = processor(images=image, return_tensors="pd")

    inputs = [inputs["pixel_values"]]
    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model, _ = paddle_trace(model, input_spec)

    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)

    outputs = model(*inputs)
