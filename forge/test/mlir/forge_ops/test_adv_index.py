# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
import forge.op
from forge import ForgeModule
from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig


class AdvIndexWrapper(ForgeModule):
    def __init__(self, name, dim=0):
        self.dim = dim
        super().__init__(name)

    def forward(self, advindex_input, indeces):
        advindex_output = forge.op.AdvIndex("", advindex_input, indeces, self.dim)
        return advindex_output


@pytest.mark.parametrize(
    "operand_shapes_dtypes, dim",
    [
        # 1D tensor
        ((((16,), torch.float32), ((2,), torch.int32)), 0),
        # 2D tensor
        ((((8, 16), torch.float32), ((2,), torch.int32)), 1),
        ((((8, 16), torch.float32), ((2,), torch.int32)), 0),
        # 3D tensor
        ((((4, 8, 16), torch.float32), ((2,), torch.int32)), 2),
        ((((4, 8, 16), torch.float32), ((2,), torch.int32)), 1),
        ((((4, 8, 16), torch.float32), ((2,), torch.int32)), 0),
        # 4D tensor
        ((((2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 3),
        ((((2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 2),
        ((((2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 1),
        ((((2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 0),
        # 5D tensor
        ((((1, 2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 4),
        ((((1, 2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 3),
        ((((1, 2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 2),
        ((((1, 2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 1),
        ((((1, 2, 4, 8, 16), torch.float32), ((2,), torch.int32)), 0),
        # Negative indexing
        ((((8, 16), torch.float32), ((2,), torch.int32)), -1),
        ((((4, 8, 16), torch.float32), ((2,), torch.int32)), -2),
        ((((2, 4, 8, 16), torch.float32), ((2,), torch.int32)), -3),
    ],
)
@pytest.mark.push
def test_adv_indexing(operand_shapes_dtypes, dim):

    # Make sure we don't go out of bounds for the dimension we're indexing
    max_int = operand_shapes_dtypes[0][0][dim] - 1

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = AdvIndexWrapper("advindex_op", dim=dim)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99)),
    )
