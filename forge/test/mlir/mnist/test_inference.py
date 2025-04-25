# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from .utils import *
import forge
import pytest
from forge.verify.verify import verify


@pytest.mark.push
@pytest.mark.functional
def test_mnist_inference(forge_property_recorder):
    inputs = [torch.rand(1, 784)]

    framework_model = MNISTLinear()

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
