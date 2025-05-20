# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from datasets import load_dataset

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.autoencoder.model_utils.conv_autoencoder import ConvAE
from test.models.pytorch.vision.autoencoder.model_utils.linear_autoencoder import (
    LinearAE,
)


@pytest.mark.nightly
@pytest.mark.xfail
def test_conv_ae_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="autoencoder", variant="conv", task=Task.IMAGE_ENCODING, source=Source.GITHUB
    )

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    framework_model = ConvAE().to(torch.bfloat16)

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

    inputs = [sample_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# pretrained weights are not provided in https://github.com/udacity/deep-learning-v2-pytorch,
# so training the model is necessary to obtain meaningful outputs.


@pytest.mark.push
@pytest.mark.nightly
def test_linear_ae_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="autoencoder",
        variant="linear",
        task=Task.IMAGE_ENCODING,
        source=Source.GITHUB,
    )

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    framework_model = LinearAE().to(torch.bfloat16)

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

    inputs = [sample_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post processing
    output_image = co_out[0].to(torch.float32).view(1, 28, 28).detach().numpy()
    save_path = "forge/test/models/pytorch/vision/autoencoder/results"
    os.makedirs(save_path, exist_ok=True)
    reconstructed_image_path = f"{save_path}/reconstructed_image.png"
    plt.imsave(reconstructed_image_path, np.squeeze(output_image), cmap="gray")
