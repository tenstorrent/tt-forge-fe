# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torchvision.transforms as transforms
from datasets import load_dataset

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.monodle.model_utils.model import CenterNet3D
import onnx


@pytest.mark.nightly
@pytest.mark.skip(reason="Fatal Python error: Floating point exception")
def test_monodle_onnx(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX, model=ModelArch.MONODLE, source=Source.TORCHVISION, task=Task.OBJECT_DETECTION
    )

    # Load data sample
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    image = next(iter(dataset.skip(10)))["image"]

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)
    inputs = [img_tensor]

    # Load Model
    torch_model = CenterNet3D(backbone="dla34")
    torch_model.eval()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/monodle.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
