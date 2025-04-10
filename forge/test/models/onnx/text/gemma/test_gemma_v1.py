# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify
from test.utils import download_model

from forge.forge_property_utils import Framework, Source, Task
import onnx
import torch


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "google/gemma-1.1-2b-it",
            marks=pytest.mark.skip(
                reason="Insufficient host DRAM to run this model; Flakey test, not hitting host OOM on every run"
            ),
        ),
        pytest.param(
            "google/gemma-1.1-7b-it",
            marks=pytest.mark.skip(reason="Skipping due to the current CI/CD pipeline limitations"),
        ),
    ],
)
def test_gemma_v1_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX, model="gemma", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")

    # Load model and tokenizer from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()
    prompt = "Write me a poem about Machine Learning."
    input = tokenizer(prompt, return_tensors="pt")
    inputs = [input["input_ids"]]

    # Export model to ONNX
    onnx_path = f"{tmp_path}/gemma_v1.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=17)

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
