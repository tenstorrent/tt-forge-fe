# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


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
        outputs = self.fpn(x)
        outputs = tuple(outputs.values())
        return outputs
