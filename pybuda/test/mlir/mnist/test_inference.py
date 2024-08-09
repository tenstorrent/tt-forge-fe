# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pybuda._C import DataFormat
from pybuda.config import _get_global_compiler_config
import torch
from torch import nn

from .utils import *

import pybuda


def test_mnist_inference():
    compiler_cfg = _get_global_compiler_config()
    df = DataFormat.Float16_b
    compiler_cfg.default_df_override = df
    compiler_cfg.default_accumulate_df = df

    inputs = [torch.rand(1, 784, dtype=torch.bfloat16)]

    framework_model = MNISTLinear()
    fw_out = framework_model(*inputs)

    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*[i.to("tt") for i in inputs])

    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]
