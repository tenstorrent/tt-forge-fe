# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import torch
import pytest
import urllib
from PIL import Image
from torchvision import transforms
import os
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name

variants = [
    pytest.param("hardnet68", id="hardnet68"),
    pytest.param("hardnet85", id="hardnet85"),
    pytest.param("hardnet68ds", id="hardnet68ds"),
    pytest.param("hardnet39ds", id="hardnet39ds"),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.skip(reason="dependent on CCM repo")
def test_hardnet_pytorch(record_forge_property, variant):
    # load only the model architecture without pre-trained weights.
    model = torch.hub.load("PingoLH/Pytorch-HarDNet", variant, pretrained=False)

    # load the weights downloaded from https://github.com/PingoLH/Pytorch-HarDNet
    checkpoint_path = f"hardnet/weights/{variant}.pth"

    # Load weights from the checkpoint file and maps tensors to CPU, ensuring compatibility even without a GPU.
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Inject weights into model
    model.load_state_dict(state_dict)
    model.eval()

    # STEP 2: Prepare input
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # Preprocessing
    input_image = Image.open(filename)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    module_name = build_module_name(framework="pt", model="hardnet", variant=variant)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)
