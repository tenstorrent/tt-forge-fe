# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer

import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.models.models_utils import build_optimum_cli_command
from test.utils import download_model
import subprocess
import onnx

variants = ["microsoft/Phi-3.5-mini-instruct"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.skip("Transient test - Out of memory due to other tests in CI pipeline")
@pytest.mark.parametrize("variant", variants)
def test_phi3_5_causal_lm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.PHI3_5,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)

    # prepare input
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    inputs = [inputs["input_ids"], inputs["attention_mask"]]

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
