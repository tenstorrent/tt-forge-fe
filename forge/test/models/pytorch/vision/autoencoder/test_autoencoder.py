# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torchvision.transforms as transforms
from datasets import load_dataset

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.autoencoder.utils.conv_autoencoder import ConvAE
from test.models.pytorch.vision.autoencoder.utils.linear_autoencoder import LinearAE
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
def test_conv_ae_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="autoencoder", variant="conv", task=Task.IMAGE_ENCODING, source=Source.GITHUB
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

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


# pretrained weights are not provided in https://github.com/udacity/deep-learning-v2-pytorch,
# so training the model is necessary to obtain meaningful outputs.


@pytest.mark.push
@pytest.mark.nightly
def test_linear_ae_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="autoencoder",
        variant="linear",
        task=Task.IMAGE_ENCODING,
        source=Source.GITHUB,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

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

    # Inference
    output = compiled_model(sample_tensor)

    # Post processing
    output_image = output[0].view(1, 28, 28).detach().numpy()
    save_path = "forge/test/models/pytorch/vision/autoencoder/results"
    os.makedirs(save_path, exist_ok=True)
    reconstructed_image_path = f"{save_path}/reconstructed_image.png"
    plt.imsave(reconstructed_image_path, np.squeeze(output_image), cmap="gray")
