# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
import onnx
from test.models.pytorch.text.nanogpt.test_nanogpt import Wrapper
from third_party.tt_forge_models.nanogpt.pytorch import ModelLoader


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize(
    "variant",
    [
        "FinancialSupport/NanoGPT",
    ],
)
def test_nanogpt_text_generation_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.NANOGPT,
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = ModelLoader()
    model = loader.load_model()
    input_tokens = loader.load_inputs()
    torch_model = Wrapper(model)
    input_ids = input_tokens["input_ids"]
    attn_mask = input_tokens["attention_mask"]
    inputs = [input_ids, attn_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1] + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
