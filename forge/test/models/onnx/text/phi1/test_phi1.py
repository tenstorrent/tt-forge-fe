# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import subprocess
import onnx
from transformers import (
    AutoTokenizer,
    PhiForCausalLM,
    PhiForSequenceClassification,
    PhiForTokenClassification,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = ["microsoft/phi-1"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi_causal_lm(forge_property_recorder, variant, tmp_path):

    # Record model details
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="phi1",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CAUSAL_LM,
    )
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load tokenizer and model from HuggingFace
    framework_model = download_model(PhiForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)

    # input_prompt
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(input_prompt, return_tensors="pt")

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

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
    subprocess.run(command, check=True, capture_output=True)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs, module_name)

    # Model Verification
    verify(
        sample_inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
