# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.squeezebert.pytorch import ModelLoader

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
import torch
import onnx


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant", [pytest.param("squeezebert/squeezebert-mnli", marks=pytest.mark.pr_models_regression)]
)
def test_squeezebert_sequence_classification_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SQUEEZEBERT,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = ModelLoader()
    torch_model = loader.load_model()
    input_tokens = loader.load_inputs()
    inputs = [input_tokens]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # post processing
    predicted_class_id = co_out[0].argmax().item()
    predicted_category = torch_model.config.id2label[predicted_class_id]
    print(f"predicted category: {predicted_category}")
