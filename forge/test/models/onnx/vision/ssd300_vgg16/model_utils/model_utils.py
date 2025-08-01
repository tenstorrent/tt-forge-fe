# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor


class SSDWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.eval()
        self.model = model

    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:

        # transform the input
        images, targets = self.model.transform(images, targets)

        # get the features from the backbone
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.model.head(features)
        output = [head_outputs["bbox_regression"], head_outputs["cls_logits"], features[0]]

        return output
