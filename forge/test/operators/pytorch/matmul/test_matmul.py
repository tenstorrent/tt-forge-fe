# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type (e.g. add, matmul, conv2d, etc.)
# 2. Operand source(s):
#    (+)  2.1 From another op
#           - Operator -> input
#    (+)  2.2 From DRAM queue
#           - Operator is first node in network
#           - Input_queue flag = false
#    (+)  2.3 Const Inputs (const eval pass)
#           - Operator where all inputs are constants.
#    (+)  2.4 From host
#           - Input tensor as input of network
#           - Operator is first node in network
#           - Input_queue flag = true
# 3. Tensor ranks:
#    (+)  3.1 Full tensor (i.e. full expected shape)
#           - 3-4 by default P1 (high prioriy)
#           - 2, 5, ++ include P2 (lower prioriy)
#    (+)  3.2 Tensor reduce on one or more dims to 1
#           - Vector
#           - Only one dim is not equal to 1
#    (-)  3.3 Scalar P2
#           - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
#    (+)  4.1 Divisible by 32
#    (+)  4.2 Prime numbers
#    (+)  4.3 Very large (thousands, 10s of thousands)
#           - 100x100, 100x1000
#           - maybe nightly only
#    (+)  4.4 Extreme ratios between height/width
#    (/)  4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
#    (/)  5.1 Output DF
#    (/)  5.2 Intermediate DF
#    (/)  5.3 Accumulation DF
#    (+)  5.4 Operand DFs
#           - Fix HiFi4 for math fidelity value
#    (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
#           - Fix fp16b (default) for data format value
#    (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
#    (+) 8. Special cases - if applicable
# 9. Variable number of operands - if applicable
#    (/) Few representative values
#    (/) Reuse inputs for selected operators


import pytest
import torch

from typing import List, Dict, Type
from loguru import logger

import forge
import forge.op

from forge.op_repo import TensorShape

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import FailingReasons
from test.operators.utils.utils import TestDevice
from test.operators.utils import ShapeUtils
from test.operators.utils import PytestParamsUtils
from test.operators.utils import ValueRanges


class ModelFromAnotherOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_another_op"

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return torch.matmul(xx, yy)


class ModelFromHost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_host"

    def forward(self, x, y):
        return torch.matmul(x, y)


class ModelFromDramQueue(ModelFromHost):
    def __init__(self):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_dram_queue"


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, shape: TensorShape):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_const_eval_pass"
        self.c1 = (torch.rand(ShapeUtils.reduce_microbatch_size(shape), requires_grad=False) - 0.5).detach()
        self.c2 = (
            torch.rand(ShapeUtils.reduce_microbatch_size(ShapeUtils.switch_last_two(shape)), requires_grad=False) - 0.5
        ).detach()

    def forward(self, x, y):
        mm1 = torch.matmul(self.c1, self.c2)
        mm2 = torch.matmul(x, y)
        aa = torch.add(mm1, mm2)
        return aa


def verify(
    test_device: TestDevice,
    model: Type[torch.nn.Module],
    input_shape: TensorShape,
    input_params: List[Dict] = [],
    input_source_flag: InputSourceFlags = None,
    dev_data_format: forge.DataFormat = None,
    math_fidelity: forge.MathFidelity = None,
):
    """Common verification function for all tests"""

    pytorch_model = model(input_shape) if model == ModelConstEvalPass else model()

    # generate shapes:
    input_shapes = [input_shape, ShapeUtils.switch_last_two(input_shape)]

    logger.trace(f"***input_shapes: {input_shapes}")

    VerifyUtils.verify(
        model=pytorch_model,
        test_device=test_device,
        input_shapes=input_shapes,
        input_params=input_params,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
        value_range=ValueRanges.SMALL,
    )


# Prepare the tests data which is list of pytest.param() objects.
# First, create the PytestParamsUtils object that holds the test data.
utils = PytestParamsUtils()

# fmt: off

# Then use generate_test_params_list method to generate the test data.
# It should be called just once to generate all test cases.

# Generate parameters for testing 3D and 4D input shapes for matmul operations
# with different model configurations and data format fidelity levels
utils.generate_test_params_list(
    PytestParamsUtils.get_shape_params(3, 4, id_name="input_shape"),    # All 3D and 4D shapes
    PytestParamsUtils.create_pytest_params(
        [
            ModelFromAnotherOp,
            ModelFromHost,
            ModelFromDramQueue,
            ModelConstEvalPass,
        ],
        id_name="model_type",
    ),
    PytestParamsUtils.get_default_df_param(id_name="dev_data_format"),
    PytestParamsUtils.get_default_mf_param(id_name="math_fidelity"),
)
utils.add_mf_test_params_list(
    PytestParamsUtils.create_pytest_params([(1, 3, 3), ], id_name="input_shape", ),
    PytestParamsUtils.create_pytest_params([ModelFromHost, ], id_name="model_type", ),
)
utils.add_df_test_params_list(
    PytestParamsUtils.create_pytest_params([(1, 3, 3), ], id_name="input_shape", ),
    PytestParamsUtils.create_pytest_params([ModelFromHost, ], id_name="model_type", ),
)

utils.extend_shape_params_with_marks(
  # (((input_shape), model_type, dev_data_format, math_fidelity), pytest.mark), wildcard = None in filtering, if used instead of pytest.mark it will clear all marks
    (((10, 10000, 1),                     None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck
    (((1, 10, 10000, 1),                  None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck
    (((9, 1, 9920, 1),                    None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck
    (((10, 10, 10000, 1),                 None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck

    (((1, 1, 23),                         None, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((2, 3, 4),            ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((11, 45, 17),         ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((11, 64, 1),          ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((100, 100, 100),      ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((10, 1000, 100),      ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((32, 32, 64),         ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((64, 160, 96),        ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((11, 17, 41),         ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           # AssertionError: PCC check failed
    (((13, 89, 3),          ModelConstEvalPass, None, None),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),             # AssertionError: If inner dimension is not the same for matmul, one of operands must hav...
    (((1, 2, 3, 4),                       None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 11, 45, 17),                    None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 11, 1, 23),                     None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 11, 64, 1),                     None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 100, 100, 100),                 None, None, None),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),             # AssertionError
    (((1, 10, 1000, 100),                 None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 1, 10, 1000),                   None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 1, 9920, 1),                    None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 11, 17, 41),                    None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((1, 13, 89, 3),                     None, None, None),  pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)),   # RuntimeError: TT_THROW @...
    (((3, 11, 45, 17),                    None, None, None),  pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),      # RuntimeError: Unhandled attribute type
    (((2, 2, 3, 4),                       None, None, None),  pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),      # RuntimeError: Unhandled attribute type
    (((4, 11, 1, 23),                     None, None, None),  pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),      # RuntimeError: Unhandled attribute type
    (((5, 11, 64, 1),                     None, None, None),  pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),      # RuntimeError: Unhandled attribute type
    (((6, 100, 100, 100),                 None, None, None),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),             # AssertionError
    (((7, 10, 1000, 100),                 None, None, None),  pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),      # RuntimeError: Unhandled attribute type
    (((8, 1, 10, 1000),                   None, None, None),  pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),      # RuntimeError: Unhandled attribute type
    (((11, 32, 32, 64),                   None, None, None),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),             # AssertionError
    (((12, 64, 160, 96),                  None, None, None),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),             # AssertionError
    (((13, 11, 17, 41),                   None, None, None),  pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)),      # RuntimeError: Unhandled attribute type
    (((14, 13, 89, 3),                    None, None, None),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),             # AssertionError

    (((1, 3, 3), None, forge.DataFormat.Int8,      forge.MathFidelity.HiFi4),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),   # AssertionError
    (((1, 3, 3), None, forge.DataFormat.RawUInt8,  forge.MathFidelity.HiFi4),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),   # AssertionError
    (((1, 3, 3), None, forge.DataFormat.RawUInt32, forge.MathFidelity.HiFi4),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),   # AssertionError
    (((1, 3, 3), None, forge.DataFormat.RawUInt16, forge.MathFidelity.HiFi4),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),   # AssertionError
    (((1, 3, 3), None, forge.DataFormat.UInt16,    forge.MathFidelity.HiFi4),  pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),   # AssertionError
)
# fmt: on


# Example of how to run just specific case(s) - first skip all, then enable what case to run:
# utils.extend_shape_params_with_marks(
# (((None), None, None, None), pytest.mark.skip(reason="Skip to execute just selected")),
# (((1, 3, 3), None, forge.DataFormat.Bfp2, None), None),
# )


@pytest.mark.parametrize(utils.test_list_fields, utils.test_list)
def test_operator(
    test_device,
    model_type,
    input_shape,
    dev_data_format,
    math_fidelity,
):

    input_source_flag = None
    if model_type == ModelFromDramQueue:
        input_source_flag = InputSourceFlags.FROM_DRAM

    verify(
        test_device=test_device,
        model=model_type,
        input_shape=input_shape,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


# Tests for special cases (by PyTorch documentation)
utils = PytestParamsUtils()
utils.set_list_fields(["shape_1", "shape_2"])
# fmt: off
utils.add_test_params(
    pytest.param((3, ), (3, ),           id="shape_1=(3, )-shape_2=(3, )"),             #
    pytest.param((7, 3), (3, 7),         id="shape_1=(7, 3)-shape_2=(3, 7)"),           #  This test passes in forge!
    pytest.param((3, ), (3, 7),          id="shape_1=(3, )-shape_2=(3, 7)"),            #
    pytest.param((32, 64), (64, ),       id="shape_1=(32, 64)-shape_2=(64, )"),         #
    pytest.param((64, ), (3, 1, 64, 32), id="shape_1=(64, )-shape_2=(3, 1, 64, 32)"),   #
)
# fmt: on

# fmt: off
utils.extend_shape_params_with_marks(
    # Matmul op when two input tensors are vectors is not supported:
    (((3, ), (3, )),           pytest.mark.xfail(reason=FailingReasons.COMPILATION_FAILED)), # tvm.error.InternalError

    # Matmul op if the first argument is 1-dimensional and the second argument is 2-dimensional is not supported:
    (((3, ), (3, 7)),          pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)), # AssertionError: Setting a tensor value of incorrect shape: (1, 7) vs torch.Size([7])

    # Matmul op if the first argument is 2-dimensional and the second argument is 1-dimensional is not suppported:
    (((32, 64), (64, )),       pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)), # RuntimeError: TT_THROW @...

    # Matmul op when one of the arguments is 1-dimensional and the other one is N-dimensional is not suppported:
    (((64, ), (3, 1, 64, 32)), pytest.mark.xfail(reason=FailingReasons.UNSUPPORTED_DIMENSION)), # AssertionError: Broadcast on dimensions beyond 3rd
                                                                                                # is not supported [[1, 1, 1, 64], [3, 1, 64, 32]] 4
)
# fmt: on


@pytest.mark.parametrize(utils.test_list_fields, utils.test_list)
def test_operator_special_cases(test_device, shape_1, shape_2):
    pytorch_model = ModelFromHost()
    input_shapes = [
        shape_1,
        shape_2,
    ]

    VerifyUtils.verify(
        model=pytorch_model,
        test_device=test_device,
        input_shapes=input_shapes,
    )
