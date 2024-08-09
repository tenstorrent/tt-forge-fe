# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from .utils import *
import pybuda

def test_mnist_inference():
    inputs = [torch.rand(1, 784)]

    framework_model = MNISTLinear()
    fw_out = framework_model(*inputs)

    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*[i.to("tt") for i in inputs])

    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]
