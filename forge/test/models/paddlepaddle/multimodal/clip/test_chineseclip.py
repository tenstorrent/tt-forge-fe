import requests
from PIL import Image

import pytest

import paddle
from paddlenlp.transformers import ChineseCLIPProcessor, ChineseCLIPTokenizer, ChineseCLIPModel, ChineseCLIPTextModel, ChineseCLIPVisionModel

from forge.tvm_calls.forge_utils import paddle_trace
import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["OFA-Sys/chinese-clip-vit-base-patch16"]

@pytest.mark.parametrize("variant", variants)
def test_chineseclip_text(variant):
    model = ChineseCLIPTextModel.from_pretrained(variant)
    model.eval()
    tokenizer = ChineseCLIPTokenizer.from_pretrained(variant)
    inputs = tokenizer("一只猫的照片", padding=True, return_tensors="pd")
    
    inputs = [inputs["input_ids"]]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)

    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)

@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail()
def test_chineseclip_vision(variant):
    model = ChineseCLIPVisionModel.from_pretrained(variant)
    model.eval()
    processor = ChineseCLIPProcessor.from_pretrained(variant)

    image = Image.open(requests.get("https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg", stream=True).raw)

    inputs = processor(images=image, return_tensors="pd")

    inputs = [inputs["pixel_values"]]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)

    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)

@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_chineseclip(variant, forge_property_recorder):
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="chineseclip",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_ENCODING,
    )

    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    model = ChineseCLIPModel.from_pretrained(variant)
    model.eval()
    processor = ChineseCLIPProcessor.from_pretrained(variant)

    text=["椅子", "玫瑰", "小火龙", "皮卡丘"]
    image = Image.open(requests.get("https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg", stream=True).raw)
    inputs = processor(images=image, text=text, return_tensors="pd", padding=True)

    inputs = [inputs["input_ids"], inputs["pixel_values"]]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)

    
    # Test framework model
    outputs = framework_model(*inputs)

    image_embed = outputs.image_embeds  
    text_embeds = outputs.text_embeds   

    image_embed = paddle.nn.functional.normalize(image_embed, axis=-1)
    text_embeds = paddle.nn.functional.normalize(text_embeds, axis=-1)

    similarities = paddle.matmul(text_embeds, image_embed.T) 
    similarities = similarities.squeeze().numpy()             

    for t, sim in zip(text, similarities):
        print(f"{t}: similarity = {sim:.4f}")

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name)
    
    # Verify
    verify(inputs, framework_model, compiled_model)




