# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
from transformers import AutoTokenizer, CodeGenForCausalLM, CodeGenModel

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)
from test.utils import download_model
from test.models.models_utils import TextModelWrapper

CodeGenModel._prepare_4d_causal_attention_mask_with_cache_position = (
    _prepare_4d_causal_attention_mask_with_cache_position
)


variants = [
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-nl",
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_codegen(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.CODEGEN,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model (with tokenizer)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = download_model(CodeGenForCausalLM.from_pretrained, variant, use_cache=False)
    model.eval()

    # Input prompt
    input_prompt = "def hello_world():"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    inputs = [input_ids, attn_mask]

    framework_model = TextModelWrapper(model=model, text_embedding=model.transformer.wte)
    framework_model.eval()

    # Export to ONNX
    variant_name = variant.replace("/", "_")
    onnx_path = f"{forge_tmp_path}/{variant_name}.onnx"
    torch.onnx.export(
        framework_model,
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

    # Compile with Forge
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
