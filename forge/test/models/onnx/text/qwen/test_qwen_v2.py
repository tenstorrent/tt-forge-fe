# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.models.models_utils import build_optimum_cli_command
import subprocess
import onnx
import torch


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "Qwen/Qwen2.5-0.5B",
            marks=pytest.mark.skip("Transient test - Out of memory due to other tests in CI pipeline"),
        ),
        pytest.param(
            "Qwen/Qwen2.5-1.5B",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB"),
        ),
        pytest.param(
            "Qwen/Qwen2.5-3B",
            marks=pytest.mark.skip(reason="Segmentation Fault"),
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",
            marks=pytest.mark.skip("Transient test - Out of memory due to other tests in CI pipeline"),
        ),
        pytest.param(
            "Qwen/Qwen2.5-1.5B-Instruct",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB"),
        ),
        pytest.param(
            "Qwen/Qwen2.5-3B-Instruct",
            marks=pytest.mark.skip(reason="Segmentation Fault"),
        ),
    ],
)
def test_qwen_clm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.QWENV2,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    framework_model = AutoModelForCausalLM.from_pretrained(variant, device_map="cpu", return_dict=False)
    framework_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(variant)

    # Prepare input
    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    inputs = [input_ids, attention_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    if variant not in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]:
        command = build_optimum_cli_command(variant, forge_tmp_path)
        subprocess.run(command, check=True)
    else:
        torch.onnx.export(framework_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
