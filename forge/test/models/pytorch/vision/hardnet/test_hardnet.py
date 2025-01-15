# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib

import pytest
import torch
from PIL import Image
from torchvision import transforms

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name

variants = [
    pytest.param("hardnet68", id="hardnet68"),
    pytest.param("hardnet85", id="hardnet85"),
    pytest.param("hardnet68ds", id="hardnet68ds"),
    pytest.param("hardnet39ds", id="hardnet39ds"),
]


@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.skip(reason="dependent on CCM repo")
def test_hardnet_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="hardnet", variant=variant)

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # load only the model architecture without pre-trained weights.
    framework_model = torch.hub.load("PingoLH/Pytorch-HarDNet", variant, pretrained=False)

    # load the weights downloaded from https://github.com/PingoLH/Pytorch-HarDNet
    checkpoint_path = f"hardnet/weights/{variant}.pth"

    # Load weights from the checkpoint file and maps tensors to CPU, ensuring compatibility even without a GPU.
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Inject weights into model
    framework_model.load_state_dict(state_dict)
    framework_model.eval()

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

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
