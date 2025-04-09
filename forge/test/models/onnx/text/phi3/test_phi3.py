# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoTokenizer,
    Phi3ForCausalLM,
)

import forge
import subprocess
import onnx
from forge.verify.verify import verify
from test.utils import download_model
from test.models.utils import Framework, Source, Task, build_module_name
from test.models.models_utils import build_optimum_cli_command

variants = ["microsoft/phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.skip(reason="Transient test - Out of memory due to other tests in CI pipeline")
def test_phi3_causal_lm_onnx(forge_property_recorder, variant, tmp_path):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX, model="phi3", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    if variant == "microsoft/phi-3-mini-4k-instruct":
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    torch_model = download_model(
        Phi3ForCausalLM.from_pretrained, variant, trust_remote_code=True, use_cache=False, return_dict=False
    )
    torch_model.eval()

    # prepare input
    input_prompt = "Africa is an emerging economy because"
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
    command = build_optimum_cli_command(variant, tmp_path)
    subprocess.run(command, check=True)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    inputs = [input_ids, attn_mask]
    compiled_model = forge.compile(
        onnx_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
