# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from forge.op.eval.common import compare_with_golden_pcc
import os


# SPDX-FileCopyrightText: Copyright (c) 2018 Udacity
#
# SPDX-License-Identifier: MIT
# https://github.com/udacity/deep-learning-v2-pytorch
class ConvAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_conv2d_1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.encoder_conv2d_2 = torch.nn.Conv2d(16, 4, 3, padding=1)
        self.encoder_max_pool2d = torch.nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder_conv2d_1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder_conv2d_2 = torch.nn.ConvTranspose2d(16, 1, 2, stride=2)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_conv2d_1(x)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)
        act = self.encoder_conv2d_2(act)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)

        # Decode
        act = self.decoder_conv2d_1(act)
        act = self.act_fun(act)
        act = self.decoder_conv2d_2(act)

        return act


class LinearAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_lin1 = torch.nn.Linear(784, 128)
        self.encoder_lin2 = torch.nn.Linear(128, 64)
        self.encoder_lin3 = torch.nn.Linear(64, 12)
        self.encoder_lin4 = torch.nn.Linear(12, 3)

        # Decoder
        self.decoder_lin1 = torch.nn.Linear(3, 12)
        self.decoder_lin2 = torch.nn.Linear(12, 64)
        self.decoder_lin3 = torch.nn.Linear(64, 128)
        self.decoder_lin4 = torch.nn.Linear(128, 784)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_lin1(x)
        act = self.act_fun(act)
        act = self.encoder_lin2(act)
        act = self.act_fun(act)
        act = self.encoder_lin3(act)
        act = self.act_fun(act)
        act = self.encoder_lin4(act)

        # Decode
        act = self.decoder_lin1(act)
        act = self.act_fun(act)
        act = self.decoder_lin2(act)
        act = self.act_fun(act)
        act = self.decoder_lin3(act)
        act = self.act_fun(act)
        act = self.decoder_lin4(act)

        return act


def test_conv_ae_pytorch(test_device):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH
    os.environ["FORGE_DISABLE_ERASE_INVERSE_OPS_PASS"] = "1"

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

    compiled_model = forge.compile(model, sample_inputs=[sample_tensor], module_name="pt_conv_ae")


def test_linear_ae_pytorch(test_device):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    os.environ["FORGE_DISABLE_ERASE_INVERSE_OPS_PASS"] = "1"

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
    compiled_model = forge.compile(model, sample_inputs=[sample_tensor], module_name="pt_linear_ae")
    co_out = compiled_model(sample_tensor)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)
