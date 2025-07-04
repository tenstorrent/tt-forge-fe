# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer, FalconForCausalLM

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import generate_no_cache, pad_inputs
from test.utils import download_model


variants = [
    pytest.param("tiiuae/Falcon3-1B-Base"),

]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_falcon_3(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.FALCON3,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.GENERALITY,
    )

    if variant != "tiiuae/Falcon3-1B-Base":
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()

    # prepare input
    input_text = "Write a function to calculate the factorial of a number"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    padded_inputs, seq_len = pad_inputs(inputs)

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/falcon3.onnx"
    torch.onnx.export(
        framework_model,
        (padded_inputs,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
    )

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Wrap and compile
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model)

    # Post-processing
    generated_text = generate_no_cache(
        max_new_tokens=50,
        model=compiled_model,
        inputs=padded_inputs,
        seq_len=seq_len,
        tokenizer=tokenizer,
    )

    print("Generated text:", generated_text)
