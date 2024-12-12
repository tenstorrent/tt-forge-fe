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

variants = [
    pytest.param("hardnet68", id="hardnet68"),
    pytest.param("hardnet85", id="hardnet85"),
    pytest.param(
        "hardnet68ds", id="hardnet68ds", marks=pytest.mark.xfail(reason="Runtime error: Invalid arguments to reshape")
    ),
    pytest.param(
        "hardnet39ds", id="hardnet39ds", marks=pytest.mark.xfail(reason="Runtime error: Invalid arguments to reshape")
    ),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.skip(reason="dependent on CCM repo")
def test_hardnet_pytorch(test_device, variant):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    if variant in ["hardnet68", "hardnet39"]:
        compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=f"pt_{variant}")
    if compiler_cfg.compile_depth == forge.CompileDepth.FULL:
        co_out = compiled_model(input_batch)

        fw_out = model(input_batch)

        co_out = [co.to("cpu") for co in co_out]
        fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

        assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
