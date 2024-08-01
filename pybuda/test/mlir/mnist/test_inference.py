# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from .utils import *


def test_mnist_inference():
    inputs = [torch.rand(1, 784)]

    framework_model = MNISTLinear()
    fw_out = framework_model(*inputs)

    compiled_model = torch.compile(framework_model.to("tt"), backend="tt")
    co_out = compiled_model(*[i.to("tt") for i in inputs])

    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]
