# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.verify.verify import verify

from .utils import *


@pytest.mark.push
def test_mnist_inference():
    inputs = [torch.rand(1, 784)]

    framework_model = MNISTLinear()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
