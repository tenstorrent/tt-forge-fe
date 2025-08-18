# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import onnx
import torch
from transformers import (
    AlbertForMaskedLM,
    AlbertTokenizer,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model

sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


params = [
    pytest.param("base", "v1"),
    pytest.param("large", "v1"),
    pytest.param("xlarge", "v1"),
    pytest.param("xxlarge", "v1"),
    pytest.param("base", "v2"),
    pytest.param("large", "v2"),
    pytest.param("xlarge", "v2"),
    pytest.param("xxlarge", "v2"),
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("size,variant", params)
def test_albert_masked_lm_onnx(size, variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.ALBERT,
        variant=f"{size}_{variant}",
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    model_ckpt = f"albert-{size}-{variant}"

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AlbertTokenizer.from_pretrained, model_ckpt)
    framework_model_pytorch = download_model(AlbertForMaskedLM.from_pretrained, model_ckpt, return_dict=False)
    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(
        framework_model_pytorch,
        tuple(inputs),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=17,
    )

    # Load ONNX model
    onnx.checker.check_model(onnx_path)
    onnx_model = onnx.load(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile ONNX model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name=module_name,
    )

    # Verify compiled model
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    predicted_token_class_ids = co_out[0].argmax(-1)
    predicted_token_class_ids = torch.masked_select(predicted_token_class_ids, input_tokens["attention_mask"][0] == 1)

    # Decode into readable text
    predicted_text = tokenizer.decode(predicted_token_class_ids.tolist(), skip_special_tokens=True)

    print(f"Context: {sample_text}")
    print(f"Answer: {predicted_text}")
