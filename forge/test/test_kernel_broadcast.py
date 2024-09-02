# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
import torch
import yaml
from .common import run


def test_kernel_broadcast_transpose(test_device):
    @run(
        verify_cfg=forge.VerifyConfig(
            test_kind=forge.verify.TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            run_net2pipe=True,
            ),
    )
    def kernel_broadcast_transpose(x, y):
        x = forge.op.Transpose("transpose", x, -2, -1)
        x = forge.op.Add("add", y, x)
        return x

    x = forge.Tensor.create_from_torch(torch.rand((1, 1, 128, 1)))
    y = forge.Tensor.create_from_torch(torch.rand((1, 1, 128, 128)))
    kernel_broadcast_transpose(x, y)


def test_lhs_matmul_zbroadcast(test_device):
    forge.config.override_op_size("mm", (2, 2))

    @run(
        verify_cfg=forge.VerifyConfig(
            test_kind=forge.verify.TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            run_net2pipe=True,
            ),
    )
    def lhs_matmul_zbroadcast(rhs, lhs=None):
        return forge.op.Matmul("mm", lhs, rhs)

    lhs = forge.Tensor.create_from_torch(torch.rand((1, 1, 128, 512)), constant=True)
    rhs = forge.Tensor.create_from_torch(torch.rand((1, 128, 512, 64)))
    lhs_matmul_zbroadcast(rhs, lhs=lhs)
