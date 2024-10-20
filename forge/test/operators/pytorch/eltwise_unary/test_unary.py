# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type (e.g. add, matmul, conv2d, etc.)
# 2. Operand source(s):
#    (-)  2.1 From another op
#           - Operator -> input
#    (-)  2.2 From DRAM queue
#           - Operator is first node in network
#           - Input_queue flag = false
#    (-)  2.3 Const Inputs (const eval pass)
#           - Operator where all inputs are constants.
#    (-)  2.4 From host
#           - Input tensor as input of network
#           - Operator is first node in network
#           - Input_queue flag = true
# 3. Tensor ranks:
#    (-)  3.1 Full tensor (i.e. full expected shape)
#           - 3-4 by default P1 (high prioriy)
#           - 2, 5, ++ include P2 (lower prioriy)
#    (-)  3.2 Tensor reduce on one or more dims to 1
#           - Vector
#           - Only one dim is not equal to 1
#    (-)  3.3 Scalar P2
#           - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
#    (-)  4.1 Divisible by 32
#    (-)  4.2 Prime numbers
#    (-)  4.3 Very large (thousands, 10s of thousands)
#           - 100x100, 100x1000
#           - maybe nightly only
#    (-)  4.4 Extreme ratios between height/width
#    (-)  4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
#    (-)  5.1 Output DF
#    (-)  5.2 Intermediate DF
#    (-)  5.3 Accumulation DF
#    (-)  5.4 Operand DFs
#           - Fix HiFi4 for math fidelity value
#    (-) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
#           - Fix fp16b (default) for data format value
#    (-) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
#    (-) 8. Special cases - if applicable
# 9. Variable number of operands - if applicable
#    (-) Few representative values
#    (-) Reuse inputs for selected operators


import pytest
import torch
import torch.nn as nn
import forge
from forge.op_repo import TensorShape
from test.operators.utils import ShapeUtils

from typing import List, Dict, Type
from loguru import logger

from test.operators.utils.utils import TestDevice
from test.operators.utils import InputSourceFlags, VerifyUtils


# Operators to test:
#  ReLU - to consider omit this operator as it differs from the rest
#  sqrt
#  reciprocal
#  sigmoid


class ModelFromAnotherOp(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_another_op"
        self.operator = operator

    def forward(self, x):
        xx = torch.add(x, x)
        return self.operator(xx)


class ModelFromHost(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_host"
        self.operator = operator

    def forward(self, x):
        return self.operator(x)


class ModelFromDramQueue(ModelFromHost):
    def __init__(self, operator):
        super().__init__(operator)
        self.testname = "Element_wise_unary_operators_test_op_src_from_dram_queue"


class ModelConstEvalPass(nn.Module):
    def __init__(self, operator, shape: TensorShape):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_const_eval_pass"
        self.operator = operator
        self.c = (torch.rand(ShapeUtils.reduce_microbatch_size(shape), requires_grad=False) - 0.5).detach()

    def forward(self, x):
        cc = self.operator(self.c)
        xx = self.operator(x)
        return torch.add(xx, cc)


def verify(
    test_device: TestDevice,
    input_operator: str,
    model: Type[torch.nn.Module],
    input_shape: TensorShape,
    input_params: List[Dict] = [],
    input_source_flag: InputSourceFlags = None,
    dev_data_format: forge.DataFormat = None,
    math_fidelity: forge.MathFidelity = None,
):

    operator = getattr(torch, input_operator)
    pytorch_model = model(operator, input_shape) if model == ModelConstEvalPass else model(operator)

    input_shapes = [input_shape]

    logger.trace(f"***input_shape: {input_shape}")

    VerifyUtils.verify(
        model=pytorch_model,
        test_device=test_device,
        input_shapes=input_shapes,
        input_params=input_params,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


# fmt: off
@pytest.mark.parametrize("input_operator", ["sqrt", "reciprocal", "sigmoid"])
@pytest.mark.parametrize("model_type", [ModelFromAnotherOp, ModelFromHost, ModelFromDramQueue, ModelConstEvalPass])
@pytest.mark.parametrize("input_shape", [(1, 3)])
@pytest.mark.parametrize("dev_data_format", [forge.DataFormat.Float16_b, ])
@pytest.mark.parametrize("math_fidelity", [forge.MathFidelity.HiFi4, ])
def test_operator(
    test_device: TestDevice,
    input_operator: str,
    model_type: Type[torch.nn.Module],
    input_shape: TensorShape,
    dev_data_format: forge.DataFormat,
    math_fidelity: forge.MathFidelity,
):

    input_source_flag = None
    if model_type == ModelFromDramQueue:
        input_source_flag = InputSourceFlags.FROM_DRAM

    verify(
        input_operator=input_operator,
        model=model_type,
        test_device=test_device,
        input_shape=input_shape,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )

    return
# fmt: on


@pytest.mark.skip
def test_temp_scratch_operator():

    input = torch.randn(40, 4, 30)

    print("\n\n\n")

    # ReLU
    m = nn.ReLU()
    relu_output = m(input)
    print(f"Input: {input} | ReLU output      : {relu_output}")

    # Sqrt
    sqrt_result = torch.sqrt(input)
    print(f"Input: {input} | Sqrt result      : {sqrt_result}")

    # Reciprocal
    reciprocal_result = torch.reciprocal(input)
    print(f"Input: {input} | Reciprocal result: {reciprocal_result}")

    # Sigmoid
    m = nn.Sigmoid()
    sigmoid_output = m(input)
    print(f"Input: {input} | Sigmoid    result: {sigmoid_output}")

    # Sigmoid #2
    sigmoid_result = torch.sigmoid(input)
    print(f"Input: {input} | Sigmoid #2 result: {sigmoid_result}")

    assert torch.equal(sigmoid_output, sigmoid_result)
