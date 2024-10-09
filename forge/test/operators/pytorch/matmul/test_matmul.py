# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (-)  2.1 From another op
#       - Operator -> input
# (-)  2.2 From DRAM queue
#       - input_queue flag = false
#       - Special case of From host? May it be triggered if the operator is not the first node of the network?
#       - Can this be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# (-)  2.3 Const Inputs (const eval pass)
#       - Operator where all inputs are constants. Does it make difference if tensor is big > L1
#       - Verification via netlists that scenario is triggered???
# (-)  2.4 From host
#       - Input tensor as input of network -> Operator is first node in network and input_queue flag = true
#       - Can this scenario be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# 3 Operand shapes type(s):
# (-)  3.1 Full tensor (i.e. full expected shape)
#       - Is 3 dims max for all ops? Ex. Conv is 3d max
# (-)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (-)  3.3 Scalar
#       - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
# (-)  4.1 Divisible by 32
# (-)  4.2 Prime numbers
# (-)  4.3 Very large (thousands, 10s of thousands)
#       - 100x100, 100x1000
#       - maybe nightly only
# (-)  4.4 Extreme ratios between height/width
#      4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
# (-)  5.1 Output DF
# (-)  5.2 Intermediate DF
# (-)  5.3 Accumulation DF
# (-)  5.4 Operand DFs
# (-) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
# (-) 7. Special attributes - if applicable.. like approx_mode for Exp, for example


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
from test.operators.utils import PytestParamsUtils

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
    

# TODO: Add ModelConstEvalPass


def verify(
    test_device: TestDevice,
    model: Type[torch.nn.Module],
    input_shape: TensorShape,
    input_params: List[Dict] = [],
    input_source_flag: InputSourceFlags = None,
    dev_data_format: forge.DataFormat = None,
    math_fidelity: forge.MathFidelity = None,
):
    '''Common verification function for all tests'''

    pytorch_model = model()
    
    # generate shapes:
    input_shapes = [input_shape, ]
    if len(input_shape) == 2:
        input_shapes.append((input_shape[1], input_shape[0]))
    elif len(input_shape) == 3:
        input_shapes.append((input_shape[0], input_shape[2], input_shape[1]))
    elif len(input_shape) == 4:
        input_shapes.append((input_shape[0], input_shape[1], input_shape[3], input_shape[2]))

    logger.trace(f"***input_shapes: {input_shapes}")

    VerifyUtils.verify(
        model=pytorch_model,
        test_device=test_device,
        input_shapes=input_shapes,
        input_params=input_params,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )

utils = PytestParamsUtils()

utils.generate_test_params_list(
    PytestParamsUtils.get_shape_params(2, 3, 4, id_name="input_shape"),
    PytestParamsUtils.create_pytest_params([ModelFromAnotherOp, ModelFromHost, ModelFromDramQueue, ], id_name="model_type"),
    PytestParamsUtils.get_default_df_param(id_name="dev_data_format"),
    PytestParamsUtils.get_default_mf_param(id_name="math_fidelity"),
).add_mf_test_params_list(
    PytestParamsUtils.create_pytest_params([(3, 3), ], id_name="input_shape"),
    PytestParamsUtils.create_pytest_params([ModelFromHost, ], id_name="model_type"),
).add_df_test_params_list(
    PytestParamsUtils.create_pytest_params([(3, 3), ], id_name="input_shape"),
    PytestParamsUtils.create_pytest_params([ModelFromHost, ], id_name="model_type"),
)

utils.extend_shape_params_with_marks(
   # ((input_shape), model_type, dev_data_format, math_fidelity)
    (((1, None),                None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),              # all 2-D shapes with micro-batch size = 1
    (((1, 1, 23),               None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),
    (((10, 10000, 1),           None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck
    (((1, 10, 10000, 1),        None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck
    (((9, 1, 9920, 1),          None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck
    (((10, 10, 10000, 1),       None, None, None), pytest.mark.skip(reason=FailingReasons.UNSUPPORTED_DIMENSION)),     # Execution stuck
    (((None, None, None, None), None, None, None), pytest.mark.xfail(reason=FailingReasons.BUGGY_SHAPE)),              # All 4-D shapes

    (((1, 32, 32, 64),          None, None, None), None),                                                              #      except this
    (((1, 64, 160, 96),         None, None, None), None),                                                              #      except this
)

# Way to run just specific cases - for development purposes
# utils.extend_shape_params_with_marks(
#     (((None), None, None, None), pytest.mark.skip(reason="Skip to execute just selected")),
#     (((10, 10000, 1), None, None, None), None),
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
