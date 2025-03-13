# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger
from PIL import Image
from torchvision import models, transforms

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="RuntimeError: tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp:50: new_volume == old_volume Invalid arguments to reshape"
)
def test_googlenet_pytorch(forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="googlenet", source=Source.TORCHVISION, task=Task.IMAGE_CLASSIFICATION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Create Forge module from PyTorch model
    # Two ways to load the same model
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    framework_model = download_model(models.googlenet, pretrained=True)
    framework_model.eval()

    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
