# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)

from test.models.pytorch.multimodal.vilt.test_vilt import generate_model_vilt_question_answering_hf_pytorch
import forge
import onnx
from forge.verify.verify import verify

variants = ["dandelin/vilt-b32-finetuned-vqa"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_question_answering_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX, model=ModelArch.VILT, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )
    torch_model, inputs, model = generate_model_vilt_question_answering_hf_pytorch(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

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

    # Post processing
    logits = co_out[0]
    idx = logits.argmax(-1).item()
    print(f"Predicted answer: {model.config.id2label[idx]}")
