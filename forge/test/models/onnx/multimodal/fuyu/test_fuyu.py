# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import FuyuProcessor, FuyuForCausalLM, FuyuConfig
from PIL import Image
import requests
import torch
import onnx
import forge

from test.models.models_utils import pad_inputs
from test.models.onnx.text.fuyu.model_utils.model_utils import generate_no_cache
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from datasets import load_dataset


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "adept/fuyu-8b",
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

    pytest.xfail(reason="Requires multi-chip support")

    # Load model and tokenizer
    config = FuyuConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = FuyuConfig(**config_dict)

    framework_model = FuyuForCausalLM.from_pretrained(variant, device_map="cpu", config=config)
    framework_model.eval()
    processor = FuyuProcessor.from_pretrained(variant, use_cache=False)

    # Prepare input
    prompt = "Generate a coco-style caption.\n"
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    image = next(iter(dataset.skip(10)))["image"]

    model_inputs = processor(images=image, text=prompt, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    image_patches = model_inputs["image_patches"]
    image_patches_indices = model_inputs["image_patches_indices"]

    # Pad input_ids and attention_mask
    padded_input_ids, seq_len = pad_inputs(input_ids, max_new_tokens)
    padded_attention_mask, _ = pad_inputs(attention_mask, max_new_tokens)

    # Updated model inputs
    model_inputs["input_ids"] = padded_input_ids
    model_inputs["attention_mask"] = padded_attention_mask

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
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    generated_text = generate_no_cache(
        max_new_tokens=512, model=compiled_model, inputs=model_inputs, seq_len=seq_len, tokenizer=processor.tokenizer
    )
    print("Generated:", generated_text)
