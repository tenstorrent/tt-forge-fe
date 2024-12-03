# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import pytest
import sys
import os
import torch
import torch.nn as nn
from forge.op.eval.common import compare_with_golden
from loguru import logger


class ForecastModule(nn.Module):
    def __init__(self):
        super(ForecastModule, self).__init__()

    def forward(self, x):
        forecast = x[:, :, :, -1:]
        return forecast


@pytest.mark.nightly
def test_index_pytorch(test_device):
    compiler_cfg = forge.config._get_global_compiler_config()

    model = ForecastModule()
    x = torch.load("x_bef_forecast_generic.pt")
    compiled_model = forge.compile(model, sample_inputs=[x], module_name="index")
    inputs = [x]
    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
