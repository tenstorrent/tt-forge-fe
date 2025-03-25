# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import random
import onnx
from transformers import AutoTokenizer, PhiForCausalLM
import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

from test.utils import download_model
import subprocess

variants = ["microsoft/phi-2"]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi2_clm_onnx(forge_property_recorder, variant, tmp_path):
    random.seed(0)

    # Record model details
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="phi2",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CAUSAL_LM,
    )
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load tokenizer and model
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    torch_model = download_model(
        PhiForCausalLM.from_pretrained, variant, trust_remote_code=True, use_cache=False, return_dict=False
    )
    torch_model.eval()

    # prepare input
    input_prompt = "Write a detailed analogy between mathematics and a lighthouse."
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    # Export model to ONNX
    onnx_path = f"{tmp_path}/model.onnx"
    command = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        variant,
        tmp_path,
        "--opset",
        "17",
        "--monolith",
        "--framework",
        "pt",
        "--trust-remote-code",
        "--task",
        "text-generation",
        "--library-name",
        "transformers",
        "--batch_size",
        "1",
        "--legacy",
    ]
    subprocess.run(command, check=True)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    inputs = [input_ids, attn_mask]
    compiled_model = forge.compile(onnx_model, inputs, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
