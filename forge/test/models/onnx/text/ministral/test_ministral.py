# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer

import forge
from forge.verify.verify import verify
from test.utils import download_model
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.models.models_utils import build_optimum_cli_command
import subprocess
import onnx

variants = ["ministral/Ministral-3b-instruct"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_ministral(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MINISTRAL,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, return_tensors="pt")

    # prepare input
    prompt = "What are the benefits of AI in healthcare?"
    input_tokens = tokenizer(prompt, return_tensors="pt")
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    command = build_optimum_cli_command(variant, forge_tmp_path)
    subprocess.run(command, check=True)

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
