# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import math
import scipy.stats
from loguru import logger


def test_clamp_equivalence():
    class ClampTensor(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = torch.tensor(torch.log(10 * torch.ones((3, 1, 1))))

        def forward(self, attn):
            logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
            return attn

    class ClampParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((3, 1, 1))))

        def forward(self, attn):
            logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
            return attn

    attn = torch.randn(64, 3, 64, 64)

    # Initialize both models
    model_tensor = ClampTensor()
    model_param = ClampParam()
    model_tensor.eval()
    model_param.eval()

    # Torch inference
    with torch.no_grad():
        torch_out_tensor = model_tensor(attn)
        torch_out_param = model_param(attn)

    # Compare PyTorch outputs
    logger.info(
        "torch.allclose(torch_out_tensor, torch_out_param)={}", torch.allclose(torch_out_tensor, torch_out_param)
    )
    correlation_coefficient, _ = scipy.stats.pearsonr(
        torch_out_tensor.detach().numpy().reshape(-1), torch_out_param.detach().numpy().reshape(-1)
    )
    logger.info("PCC = {}", correlation_coefficient)
    logger.info("torch_out_tensor={}", torch_out_tensor)
    logger.info("torch_out_param={}", torch_out_param)
