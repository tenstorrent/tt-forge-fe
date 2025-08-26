# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of element-wise nary operators
#

import forge.tensor
import pytest

import torch

import forge
import forge.op
from forge import ForgeModule, Tensor, DeprecatedVerifyConfig
from test.common import run
from forge.verify import TestKind, verify_module

verify_cfg = DeprecatedVerifyConfig(run_golden=True, run_net2pipe=True)  # Run backend golden check on all tests in here


@pytest.mark.parametrize("dim", [1, 2, -1])
@pytest.mark.parametrize("aligned", [True, False])
def test_concat(test_kind, test_device, dim, aligned):
    @run(
        DeprecatedVerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
    )
    def simple_concat(a, b):
        return forge.op.Concatenate("", a, b, axis=dim)

    if aligned:
        shapes = {
            -1: (1, 3, 128, 96),
            2: (1, 3, 1024, 32),
            1: (1, 1, 128, 32),
        }
        a = Tensor.create_from_torch(torch.randn((1, 3, 128, 32), requires_grad=test_kind.is_training()))
    else:
        shapes = {
            -1: (1, 3, 128, 6),
            2: (1, 3, 128, 6),
            1: (1, 1, 128, 6),
        }
        a = Tensor.create_from_torch(torch.randn((1, 3, 128, 6), requires_grad=test_kind.is_training()))
    b = Tensor.create_from_torch(torch.randn(shapes[dim], requires_grad=test_kind.is_training()))
    c = simple_concat(a, b)
