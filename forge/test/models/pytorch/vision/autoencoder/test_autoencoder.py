# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from forge.verify.compare import compare_with_golden
import os
import pytest
from test.models.pytorch.vision.autoencoder.utils.conv_autoencoder import ConvAE
from test.models.pytorch.vision.autoencoder.utils.linear_autoencoder import LinearAE
from test.models.utils import build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_conv_ae_pytorch(test_device):
    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    model = ConvAE()

    # Define transform to normalize data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load sample from MNIST dataset
    dataset = load_dataset("mnist")
    sample = dataset["train"][0]["image"]
    sample_tensor = transform(sample).unsqueeze(0)

    module_name = build_module_name(framework="pt", model="conv_autoencoder")
    compiled_model = forge.compile(model, sample_inputs=[sample_tensor], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_linear_ae_pytorch(test_device):
    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    model = LinearAE()

    # Define transform to normalize data
    transform = transforms.Compose(
        [
            transforms.Resize((1, 784)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load sample from MNIST dataset
    dataset = load_dataset("mnist")
    sample = dataset["train"][0]["image"]
    sample_tensor = transform(sample).squeeze(0)

    # Sanity
    fw_out = model(sample_tensor)

    # Inference
    module_name = build_module_name(framework="pt", model="linear_autoencoder")
    compiled_model = forge.compile(model, sample_inputs=[sample_tensor], module_name=module_name)
    co_out = compiled_model(sample_tensor)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden(golden=fw_out, calculated=co_out[0], pcc=0.99)
