# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify
from test.utils import download_model
from test.models.utils import Framework, Source, Task, build_module_name
from test.models.models_utils import build_optimum_cli_command
import subprocess
import onnx

variants = ["ministral/Ministral-3b-instruct"]


@pytest.mark.nightly
@pytest.mark.skip(reason="Out of memory")
@pytest.mark.parametrize("variant", variants)
def test_ministral(forge_property_recorder, variant, tmp_path):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="ministral",
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    # Load tokenizer and model
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, return_tensors="pt")
    torch_model = download_model(AutoModelForCausalLM.from_pretrained, variant, use_cache=False, return_dict=False)
    torch_model.eval()

    # prepare input
    prompt = "What are the benefits of AI in healthcare?"
    input_tokens = tokenizer(prompt, return_tensors="pt")
    inputs = [input_tokens["input_ids"]]

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
