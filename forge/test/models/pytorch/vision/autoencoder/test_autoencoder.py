# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torchvision.transforms as transforms
from datasets import load_dataset

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.autoencoder.utils.conv_autoencoder import ConvAE
from test.models.pytorch.vision.autoencoder.utils.linear_autoencoder import LinearAE
from test.models.utils import Framework, build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_conv_ae_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="conv_autoencoder")

    record_forge_property("module_name", module_name)

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    framework_model = ConvAE()

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

    inputs = [sample_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_linear_ae_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="linear_autoencoder")

    record_forge_property("module_name", module_name)

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    framework_model = LinearAE()

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

    inputs = [sample_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
