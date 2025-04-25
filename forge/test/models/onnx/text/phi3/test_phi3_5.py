# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task
from test.models.models_utils import build_optimum_cli_command
from test.utils import download_model
import subprocess
import onnx

variants = ["microsoft/Phi-3.5-mini-instruct"]


@pytest.mark.nightly
@pytest.mark.skip("Transient test - Out of memory due to other tests in CI pipeline")
@pytest.mark.parametrize("variant", variants)
def test_phi3_5_causal_lm_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX, model="phi3_5", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P2")

    # Load model and tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(
        AutoModelForCausalLM.from_pretrained, variant, return_dict=False, trust_remote_code=True, use_cache=False
    )
    framework_model.eval()

    # prepare input
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )
    inputs = [inputs["input_ids"], inputs["attention_mask"]]

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
