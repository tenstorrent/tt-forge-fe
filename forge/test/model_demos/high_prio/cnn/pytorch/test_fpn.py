# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import os
import forge
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class FPNWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        self.fpn = model.backbone.fpn

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
    model = FPNWrapper()

    feat0 = torch.rand(1, 256, 64, 64)
    feat1 = torch.rand(1, 512, 16, 16)
    feat2 = torch.rand(1, 2048, 8, 8)

    inputs = [feat0, feat1, feat2]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_fpn")
