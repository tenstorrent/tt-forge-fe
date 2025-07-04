# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

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
from test.models.pytorch.multimodal.llava.model_utils.utils import load_inputs
from test.models.pytorch.multimodal.llava.test_llava import load_model

variants = ["llava-hf/llava-1.5-7b-hf"]


@pytest.mark.nightly
@pytest.mark.skip(reason="Hangs at generate initial graph stage.")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_llava_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.LLAVA,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
    )

    torch_model, processor = load_model(variant)
    image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    text = "What’s shown in this image?"

    # Input sample
    input_ids, attn_mask, pixel_values = load_inputs(image, text, processor)
    inputs = [input_ids, attn_mask, pixel_values]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1], inputs[2]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
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
