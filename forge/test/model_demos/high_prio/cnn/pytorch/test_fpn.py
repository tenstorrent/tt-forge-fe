# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import forge
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict


class FPNWrapper(nn.Module):
    def __init__(self, in_channels_list, out_channels, extra_blocks=None, norm_layer=None):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks, norm_layer)

    def forward(self, feat0, feat1, feat2):
        x = OrderedDict()
        x["feat0"] = feat0
        x["feat1"] = feat1
        x["feat2"] = feat2
        return self.fpn(x)


def test_fpn_pytorch(test_device):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load FPN model
    model = FPNWrapper([10, 20, 30], 5)

    feat0 = torch.rand(1, 10, 64, 64)
    feat1 = torch.rand(1, 20, 16, 16)
    feat2 = torch.rand(1, 30, 8, 8)

    inputs = [feat0, feat1, feat2]
    compiled_model = forge.compile(model, sample_inputs=inputs)
