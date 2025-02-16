# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys

import torch
import torch.nn as nn
from loguru import logger
from mmcv.cnn import build_conv_layer

sys.path.append("forge/test/models/pytorch/vision/")
from dcnv2_custom_implementation import ModulatedDeformConv2dPack_custom


def test_dcnv2():

    # Org implementation

    class DCNV2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2 = build_conv_layer(
                {"type": "DCNv2", "deform_groups": 1},  # dcn_cfg
                256,  # planes
                256,  # planes
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            )

        def forward(self, ip):

            out = self.conv2(ip)
            return out

    model = DCNV2()
    inputs = [torch.randn(6, 256, 32, 88)]

    with torch.no_grad():
        cpu_op1 = model(inputs[0])

    # Extract weights from model
    shared_weight = model.conv2.weight.clone()

    # Custom implementation

    class DCNV2_Custom(nn.Module):
        def __init__(self):
            super(DCNV2_Custom, self).__init__()

            self.deform_conv = ModulatedDeformConv2dPack_custom(
                256, 256, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.deform_conv.weight = nn.Parameter(shared_weight)

        def forward(self, x):

            return self.deform_conv(x)

    logger.info("org implementation output={}", cpu_op1)

    model2 = DCNV2_Custom()

    with torch.no_grad():
        cpu_op2 = model2(inputs[0])

    logger.info("custom implementation output={}", cpu_op2)

    # Compare the outputs
    are_close = torch.allclose(cpu_op1, cpu_op2)
    logger.info("Are the outputs close? {}", are_close)
