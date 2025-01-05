# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from torchvision import models, transforms
from test.utils import download_model
import forge
from PIL import Image
from loguru import logger
from test.models.utils import build_module_name, Framework


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_googlenet_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="googlenet")

    record_forge_property("module_name", module_name)

    # Create Forge module from PyTorch model
    # Two ways to load the same model
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model = download_model(models.googlenet, pretrained=True)
    model.eval()

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
    input_batch_list = [input_batch]
    compiled_model = forge.compile(model, sample_inputs=input_batch_list, module_name=module_name)
