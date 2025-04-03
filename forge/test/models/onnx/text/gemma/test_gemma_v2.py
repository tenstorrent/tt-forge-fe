# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task
from test.utils import download_model
import onnx


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "google/gemma-2-2b-it",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model"),
        ),
        pytest.param(
            "google/gemma-2-9b-it",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model"),
        ),
    ],
)
def test_gemma_v2_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX, model="gemma", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")

    # Load tokenizer and model
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()
    prompt = "Write me a poem about Machine Learning."
    input = tokenizer(prompt, return_tensors="pt")
    inputs = [input["input_ids"]]

    # Export model to ONNX
    onnx_path = f"{tmp_path}/gemma_v2.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(
        framework_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
