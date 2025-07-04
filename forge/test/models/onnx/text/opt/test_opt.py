# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch 
import onnx
from transformers import (
    AutoTokenizer,
    OPTForQuestionAnswering,
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

variants = [
    # "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
]

@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_opt_qa(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.OPT, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(OPTForQuestionAnswering.from_pretrained, variant, torchscript=True)

    # Load data sample
    question, context = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant.replace('/', '_')}.onnx"
    torch.onnx.export(
        framework_model,
        tuple(inputs),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["start_logits", "end_logits"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
        },
    )

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Forge ONNX wrapper
    onnx_module = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile with Forge
    compiled_model = forge.compile(onnx_module, sample_inputs=inputs, module_name=module_name)

    # Verify output
    verify(inputs, onnx_module, compiled_model)
