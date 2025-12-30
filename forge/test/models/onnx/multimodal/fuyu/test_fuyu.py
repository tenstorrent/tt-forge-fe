# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from third_party.tt_forge_models.fuyu.pytorch.loader import ModelLoader, ModelVariant
import forge


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(ModelVariant.FUYU_8B),
    ],
)
def test_fuyu_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.FUYU,
        variant=variant.value,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs using ModelLoader
    loader = ModelLoader(variant)
    framework_model = loader.load_model()
    model_inputs = loader.load_inputs()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    torch.onnx.export(
        framework_model,
        (model_inputs),
        onnx_path,
        opset_version=17,
        input_names=["input_embeds"],
        output_names=["output"],
    )

    # Load and compile ONNX model
    onnx_model = onnx.load(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)
    compiled_model = forge.compile(framework_model, [model_inputs], module_name=module_name)

    # Model verification
    verify([model_inputs], framework_model, compiled_model)
