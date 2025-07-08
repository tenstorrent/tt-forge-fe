# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import torch
import onnx
import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties


@pytest.mark.out_of_memory
@pytest.mark.skip(reason="Skipping due to CI/CD Limitations")
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "adept/fuyu-8b",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model"),
        ),
    ],
)
def test_fuyu_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.FUYU,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    framework_model = FuyuForCausalLM.from_pretrained(variant, device_map="cpu", return_dict=False, use_cache=False)
    framework_model.eval()
    processor = FuyuProcessor.from_pretrained(variant, use_cache=False)

    # Prepare input
    prompt = "Generate a coco-style caption.\n"
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    model_inputs = processor(images=image, text=prompt, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    image_patches = model_inputs["image_patches"]
    image_patches_indices = model_inputs["image_patches_indices"]

    inputs = [input_ids, image_patches, image_patches_indices, attention_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    torch.onnx.export(
        framework_model,
        (inputs[0], inputs[1], inputs[2], inputs[3]),
        onnx_path,
        opset_version=17,
        input_names=["input_ids", "image_patches", "image_patches_indices", "attention_mask"],
        output_names=["output"],
    )

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
