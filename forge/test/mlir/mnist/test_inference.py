# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# Local Imports
import forge
from forge.verify.config import VerifyConfig
from forge.verify.verify import verify

from .utils import *


@pytest.mark.push
def test_mnist_inference():
    inputs = [torch.rand(1, 784)]

    framework_model = MNISTLinear()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_allclose=False))
