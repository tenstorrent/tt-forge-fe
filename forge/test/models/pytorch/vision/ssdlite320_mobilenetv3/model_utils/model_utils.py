# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_batch):
        outputs = self.model(input_batch)
        outputs = [outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]]
        return outputs
