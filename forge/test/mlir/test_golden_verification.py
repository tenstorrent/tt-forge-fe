# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify, verify_backward, DepricatedVerifyConfig
from forge.config import CompileDepth
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("outer_dim_x", [7])
@pytest.mark.parametrize("outer_dim_y", [7])
@pytest.mark.parametrize("inner_dim", [64])
@pytest.mark.push
def test_matmul_and_add(batch_size, outer_dim_x, outer_dim_y, inner_dim):
    class MatmulAdd(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z, t):
            mm = torch.matmul(x, y)
            add_1 = mm + z
            return add_1 + t

    inputs = [
        torch.rand(batch_size, outer_dim_x, inner_dim),
        torch.rand(batch_size, inner_dim, outer_dim_y),
        torch.rand(batch_size, outer_dim_x, outer_dim_y),
        torch.rand(batch_size, outer_dim_x, outer_dim_y),
    ]

    framework_model = MatmulAdd()
    verify_cfg = DepricatedVerifyConfig()
    verify_cfg.verify_all = True
    verify_cfg.enable_op_level_comparision = True
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, verify_cfg=verify_cfg)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("lhs", [7])
@pytest.mark.parametrize("rhs", [7])
@pytest.mark.push
def test_constant_add(batch_size, lhs, rhs):
    class ConstAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.const_1 = torch.rand(batch_size, lhs, rhs)

        def forward(self, x):
            out = x + self.const_1
            return out

    inputs = [torch.rand(batch_size, lhs, rhs)]

    framework_model = ConstAdd()

    verify_cfg = DepricatedVerifyConfig()
    verify_cfg.verify_all = True
    verify_cfg.enable_op_level_comparision = True
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, verify_cfg=verify_cfg)

    verify(inputs, framework_model, compiled_model)


compile_depths_to_test = [
    "ALL",  # this will activate verify_all = True
    *[depth for depth in CompileDepth if depth.value <= CompileDepth.SPLIT_GRAPH.value],
]


@pytest.mark.parametrize("verify_stage", compile_depths_to_test, ids=lambda x: x if isinstance(x, str) else x.name)
@pytest.mark.parametrize(
    "shapes, train",
    [
        (((1, 11, 2048), (2048, 128256), (128256, 2048)), True),
    ],
)
@pytest.mark.push
def test_matmuls(forge_property_recorder, shapes, train, verify_stage):
    shape1, shape2, shape3 = shapes

    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()
            self.rhs_param = nn.Parameter(torch.rand(shape3), requires_grad=train)

        def forward(self, x, y):
            intermediate = torch.matmul(x, y)
            return torch.matmul(intermediate, self.rhs_param)

    inputs = [
        torch.rand(shape1, requires_grad=train),
        torch.rand(shape2, requires_grad=train),
    ]

    framework_model = Matmul()
    framework_model.train() if train else framework_model.eval()

    verify_cfg = DepricatedVerifyConfig()
    if verify_stage == "ALL":
        verify_cfg.verify_all = True
    else:
        verify_cfg.stages_for_intermediate_verification = {verify_stage}
    verify_cfg.enable_op_level_comparision = True

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        training=train,
        forge_property_handler=forge_property_recorder,
        verify_cfg=verify_cfg,
    )
