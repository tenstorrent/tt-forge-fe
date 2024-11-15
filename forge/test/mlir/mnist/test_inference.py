# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from .utils import *
import forge
import pytest
from forge.verify.verify import verify


@pytest.mark.push
def test_mnist_inference():
    inputs = [torch.rand(1, 784)]

    framework_model = MNISTLinear()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs=inputs, compiled_model=compiled_model, framework_model=framework_model)
