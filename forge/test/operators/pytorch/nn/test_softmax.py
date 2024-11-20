# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing of nn operators
#
# In this test we use pytorch tensors and operators to verify forge operators


# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (+)  2.1 From another op
#       - Operator -> input
# (+)  2.2 From DRAM queue
#       - input_queue flag = false
#       - Special case of From host? May it be triggered if the operator is not the first node of the network?
#       - Can this be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# (+)  2.3 Const Inputs (const eval pass)
#       - Operator where all inputs are constants. Does it make difference if tensor is big > L1
#       - Verification via netlists that scenario is triggered???
# (+)  2.4 From host
#       - Input tensor as input of network -> Operator is first node in network and input_queue flag = true
#       - Can this scenario be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# 3 Operand shapes type(s):
# (+)  3.1 Full tensor (i.e. full expected shape)
#       - Is 3 dims max for all ops? Ex. Conv is 3d max
# (+)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (+)  3.3 Scalar
#       - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
# (+)  4.1 Divisible by 32
# (+)  4.2 Prime numbers
# (+)  4.3 Very large (thousands, 10s of thousands)
#       - 100x100, 100x1000
#       - maybe nightly only
# (+)  4.4 Extreme ratios between height/width
#      4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
# (/)  5.1 Output DF
# (/)  5.2 Intermediate DF
# (/)  5.3 Accumulation DF
# (+)  5.4 Operand DFs
# (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
# (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example


import pytest
import torch

from typing import List, Dict, Type
from loguru import logger

import forge
import forge.op

from forge.op_repo import TensorShape

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.operators.utils import FailingReasons
from test.operators.utils.utils import TestDevice
from test.operators.utils import PytestParamsUtils
from test.operators.utils import ValueRanges


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, shape, dim):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Softmax_operator_test_op_src_from_another_op"
        self.shape = shape
        self.dim = dim
        self.softmax = torch.nn.Softmax(dim=self.dim)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create operand which is input for the Softmax operator
        xx = torch.add(x, x)
        return self.softmax(xx)


class ModelFromHost(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, shape, dim):
        super(ModelFromHost, self).__init__()
        self.testname = "Softmax_operator_test_op_src_from_host"
        self.shape = shape
        self.dim = dim
        self.softmax = torch.nn.Softmax(dim=self.dim)

    def forward(self, x):
        return self.softmax(x)


class ModelFromDramQueue(torch.nn.Module):

    model_name = "model_op_src_from_dram_queue"

    def __init__(self, shape, dim):
        super(ModelFromDramQueue, self).__init__()
        self.testname = "Softmax_operator_test_op_src_from_dram_queue"
        self.shape = shape
        self.dim = dim
        self.softmax = torch.nn.Softmax(dim=self.dim)

    def forward(self, x):
        return self.softmax(x)


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, shape, dim):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Softmax_operator_test_op_src_const_eval_pass"
        self.shape = shape
        self.dim = dim
        self.softmax = torch.nn.Softmax(dim=self.dim)

        self.constant_shape = ShapeUtils.reduce_microbatch_size(shape)

        self.c = (torch.rand(*self.constant_shape, requires_grad=False) - 0.5).detach()

    def forward(self, x):
        v1 = self.softmax(self.c)
        v2 = torch.add(x, v1)
        return v2


def verify(
    test_device: TestDevice,
    model: Type[torch.nn.Module],
    dim: int,
    input_shape: TensorShape,
    number_of_operands: int,
    input_params: List[Dict] = [],
    input_source_flag: InputSourceFlags = None,
    dev_data_format: forge.DataFormat = None,
    math_fidelity: forge.MathFidelity = None,
):
    """Common verification function for all models"""

    pytorch_model = model(shape=input_shape, dim=dim)
    input_shapes = tuple([input_shape for _ in range(number_of_operands)])
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


# PREPARE TEST PARAMETERS LIST:

utils = PytestParamsUtils()

# fmt: off
utils.generate_test_params_list(
    [
        *PytestParamsUtils.join_two_params_lists(PytestParamsUtils.get_shape_params(2), PytestParamsUtils.create_pytest_params([0, 1], id_name="dim")),
        *PytestParamsUtils.join_two_params_lists(PytestParamsUtils.get_shape_params(3), PytestParamsUtils.create_pytest_params([0, 1, 2], id_name="dim")),
        *PytestParamsUtils.join_two_params_lists(PytestParamsUtils.get_shape_params(4), PytestParamsUtils.create_pytest_params([0, 1, 2, 3], id_name="dim")),
    ],
    utils.create_pytest_params([ModelFromAnotherOp, ModelFromDramQueue, ModelFromHost, ModelConstEvalPass, ], "model_type"),
    PytestParamsUtils.get_default_df_param(),
    PytestParamsUtils.get_default_mf_param(),

)
utils.add_mf_test_params_list(
    PytestParamsUtils.create_pytest_params([(1, 3, 3), ], id_name="shape", ),
    PytestParamsUtils.create_pytest_params([1, ], id_name="dim", ),
    PytestParamsUtils.create_pytest_params([ModelFromAnotherOp, ], id_name="model_type", ),
)
utils.add_df_test_params_list(
    PytestParamsUtils.create_pytest_params([(1, 3, 3), ], id_name="shape", ),
    PytestParamsUtils.create_pytest_params([1, ], id_name="dim", ),
    PytestParamsUtils.create_pytest_params([ModelFromAnotherOp, ], id_name="model_type", ),
)

utils.extend_shape_params_with_marks(
 #   ((shape),               dim,          model_type,      df,   mf)
    (((1, 1),               None,                   None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((3, 4),                  0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((45, 17),                0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((100, 100),              0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1000, 100),             0, [ModelFromHost,
                                   ModelFromDramQueue,
                                   ModelFromAnotherOp  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((10, 1000),              0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((32, 64),                0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((160, 96),               0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((17, 41),                0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((89, 3),                 0, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 3, 4),               1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 45, 17),             1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 100, 100),           1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 1000, 100),          1, [ModelFromDramQueue,
                                   ModelFromHost,
                                   ModelFromAnotherOp  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 10, 1000),           1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 32, 64),             1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 17, 41),             1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 89, 3),              1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((2, 3, 4),            None, [ModelConstEvalPass, ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((2, 3, 4),               1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 45, 17),            0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 45, 17),            1,                   None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 45, 17),            2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 1, 23),             0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 1, 23),             1, [ModelConstEvalPass, ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 1, 23),             2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 64, 1),             0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 64, 1),             1,                   None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 64, 1),             2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((100, 100, 100),         0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((100, 100, 100),         1,                   None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((100, 100, 100),         2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((10, 1000, 100),         0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((10, 1000, 100),         1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost,
                                   ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((10, 1000, 100),         2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((10, 10000, 1),          0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((10, 10000, 1),          1, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((10, 10000, 1),          2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((32, 32, 64),            0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((32, 32, 64),            1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((64, 160, 96),           1, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 17, 41),            0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 17, 41),            1,                   None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 17, 41),            2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((13, 89, 3),             0, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((13, 89, 3),             1,                   None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((13, 89, 3),             2, [ModelConstEvalPass  ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 2, 3, 4),            2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 11, 45, 17),         2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 11, 64, 1),          2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost ],       None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 100, 100, 100),      2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 10, 1000, 100),      2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost ],       None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 1, 10, 1000),        2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost ],       None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 32, 32, 64),         2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost ],       None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 64, 160, 96),        2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost ],       None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 11, 17, 41),         2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((1, 13, 89, 3),          2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((3, 11, 45, 17),         2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((2, 2, 3, 4),            2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((5, 11, 64, 1),          2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((6, 100, 100, 100),      2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((7, 10, 1000, 100),      2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((8, 1, 10, 1000),        2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((11, 32, 32, 64),        2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((12, 64, 160, 96),       2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((13, 11, 17, 41),        2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed
    (((14, 13, 89, 3),         2, [ModelFromAnotherOp,
                                   ModelFromDramQueue,
                                   ModelFromHost       ], None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)), # AssertionError: PCC check failed

    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Float16_b, forge.MathFidelity.LoFi), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),  # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi2), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi3), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Bfp2,      forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Bfp2_b,    forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Bfp4,      forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Bfp4_b,    forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Bfp8,      forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Bfp8_b,    forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Float16,   forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Float32,   forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Lf8,       forge.MathFidelity.HiFi4), pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)), # AssertionError: PCC check failed

    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.Int8,      None), pytest.mark.xfail(reason=FailingReasons.NOT_IMPLEMENTED)), # RuntimeError: "softmax_kernel_impl" not implemented for 'Int'
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.RawUInt16, None), pytest.mark.xfail(reason=FailingReasons.NOT_IMPLEMENTED)), # RuntimeError: "softmax_kernel_impl" not implemented for 'Int'
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.RawUInt32, None), pytest.mark.xfail(reason=FailingReasons.NOT_IMPLEMENTED)), # RuntimeError: "softmax_kernel_impl" not implemented for 'Int'
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.RawUInt8,  None), pytest.mark.xfail(reason=FailingReasons.NOT_IMPLEMENTED)), # RuntimeError: "softmax_kernel_impl" not implemented for 'Int'
    (((1, 3, 3), 1, ModelFromAnotherOp, forge.DataFormat.UInt16,    None), pytest.mark.xfail(reason=FailingReasons.NOT_IMPLEMENTED)), # RuntimeError: "softmax_kernel_impl" not implemented for 'Int'
)
# fmt: on


# TEST(S):
@pytest.mark.parametrize("input_shape, dim, model_type, dev_data_format, math_fidelity", utils.test_list)
def test_softmax(test_device, model_type, dim, input_shape, dev_data_format, math_fidelity):

    input_source_flag = None
    if model_type == ModelFromDramQueue:
        input_source_flag = InputSourceFlags.FROM_DRAM

    verify(
        test_device=test_device,
        model=model_type,
        dim=dim,
        input_shape=input_shape,
        number_of_operands=1,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )

    # netlist validations are skipped for now - there are no netlists support yet.


# fmt: off
def get_test_params_sortmax_inconsistency():
    params = [
        # pytest.param((32, 64),   1, ModelFromAnotherOp, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id="(32, 64)-dim=1-model_type=ModelFromAnotherOp-df=Float16_b-mf=HiFi4"),
        # pytest.param((15, 4, 3), 0, ModelFromAnotherOp, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id="(15, 4 ,3)-dim=0-model_type=ModelFromAnotherOp-df=Float16_b-mf=HiFi4"),
        pytest.param((1, 160, 96), 0, ModelFromDramQueue, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id="(1, 160, 96)-dim=0-model_type=ModelFromDramQueue-df=Float16_b-mf=HiFi4"),
        pytest.param((1, 160, 96), 1, ModelFromDramQueue, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id="(1, 160, 96)-dim=1-model_type=ModelFromDramQueue-df=Float16_b-mf=HiFi4"),
        pytest.param((1, 160, 96), 2, ModelFromDramQueue, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id="(1, 160, 96)-dim=2-model_type=ModelFromDramQueue-df=Float16_b-mf=HiFi4"),
    ]
    # params.reverse()      # COMMENT/UNCOMMENT THIS LINE TO CHANGE ORDER OF TESTS - test results inconsistency issue
    # print("\n\n\nPARAMETERS:\n\n")
    # for item in params:
    #     print(item.id)
    # print("\n")
    return params
# fmt: on


@pytest.mark.skip(reason="This test is used to reproduce test results inconsistency")
@pytest.mark.parametrize(
    "input_shape, dim, model_type, dev_data_format, math_fidelity", get_test_params_sortmax_inconsistency()
)
def test_softmax_inconsistency(test_device, model_type, dim, input_shape, dev_data_format, math_fidelity):
    """Test for checking inconsistency between dev_data_format and math_fidelity"""

    verify(
        test_device=test_device,
        model=model_type,
        dim=dim,
        input_shape=input_shape,
        number_of_operands=1,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )


# Test function for running test with specific parameters
@pytest.mark.skip(reason="This test is used to reproduce single test case")
def test_softmax_single_params(test_device, softmax_model, softmax_input_shape_dim, df, mf):
    model_type = eval(softmax_model)
    input_shape, dim = softmax_input_shape_dim
    dev_data_format = eval(f"forge.DataFormat.{df}")
    math_fidelity = eval(f"forge.MathFidelity.{mf}")
    test_softmax(test_device, model_type, dim, input_shape, dev_data_format, math_fidelity)
