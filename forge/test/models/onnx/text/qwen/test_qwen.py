# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch  
import onnx
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "Qwen/Qwen1.5-0.5B",
g        ),
    ],
)
def test_qwen1_5_causal_lm(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.QWEN15,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Setup model configuration
    config = Qwen2Config.from_pretrained(variant)
    config.use_cache = False
    config.return_dict = False

    # Load model and tokenizer with config
    framework_model = Qwen2ForCausalLM.from_pretrained(variant, config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained(variant)
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    framework_model._supports_cache_class = False

    # Example usage
    batch_size = 1
    prompt = ["My name is Jim Keller and"] * batch_size

    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant.replace('/', '_')}.onnx"
    torch.onnx.export(
        framework_model,
        inputs["input_ids"],
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq_len"}},
        opset_version=17,
    )

    # Validate ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Forge ONNX wrapper and compile
    onnx_module = forge.OnnxModule(module_name, onnx_model, onnx_path)
    compiled_model = forge.compile(onnx_module, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, onnx_module, compiled_model)
