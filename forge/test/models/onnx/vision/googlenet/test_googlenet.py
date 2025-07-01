# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import transforms

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.xfail
def test_googlenet_onnx(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GOOGLENET,
        variant="googlenet",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Download ONNX model
    model_url = (
        "https://github.com/onnx/models/raw/main/Computer_Vision/googlenet_Opset17_torch_hub/googlenet_Opset17.onnx"
    )
    onnx_path = get_file(model_url)

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Image preprocessing
    try:
        image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        file_path = get_file(image_url)
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)

    inputs = [input_batch]

    # Compile with Forge
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
