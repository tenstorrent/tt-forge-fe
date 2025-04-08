# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of reduce operators
#
# In this test we test pytorch reduce operators

# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (+)  2.1 From another op
#       - Operator -> input
# (+)  2.2 From DRAM queue - removed from test plan
#       - Operator is first node in network
#       - Input_queue flag = false
# (+)  2.3 Const Inputs (const eval pass)
#       - Operator where all inputs are constants.
# (+)  2.4 From host
#       - Input tensor as input of network
#       - Operator is first node in network
#       - Input_queue flag = true
# 3 Operand shapes type(s):
# (+)  3.1 Full tensor (i.e. full expected shape)
#       - 3-4 by default P1 (high prioriy)
#       - 2, 5, ++ include P2 (lower prioriy)
# (+)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (+)  3.3 Scalar P2
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
#       - Fix HiFi4 for math fidelity value
# (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
#       - Fix fp16b (default) for data format value
# (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
# (/) 8. Special cases - if applicable
# 9. Variable number of operands - if applicable
# (/) Few representative values
# (/) Reuse inputs for selected operators


import pytest

from typing import List, Dict, Type, Optional, Any
from loguru import logger

import random
import torch
import forge
import forge.op
import os

from forge.op_repo import TensorShape

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestPlanUtils
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Reduce_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the reduce operator
        add1 = torch.add(x, x)
        output = self.operator(add1, **self.kwargs)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Reduce_pytorch_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        output = self.operator(x, **self.kwargs)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Reduce_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.constant_shape = ShapeUtils.reduce_microbatch_size(shape)
        self.kwargs = kwargs

        self.c1 = torch.rand(*self.constant_shape) - 0.5

    def forward(self, x):
        v1 = self.operator(self.c1, **self.kwargs)
        v2 = self.operator(x, **self.kwargs)
        # add consume inputs
        add = torch.add(v1, v2)
        return add


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        number_of_operands: int = 1,
        input_params: List[Dict] = [],
        warm_reset: bool = False,
    ):
        """Common verification function for all tests"""

        operator = getattr(torch, test_vector.operator)

        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = model_type(
            operator=operator,
            opname=test_vector.operator,
            shape=test_vector.input_shape,
            kwargs=kwargs,
        )

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            # Old behavior when dev_data_format was not set
            value_range=None if test_vector.dev_data_format is not None else ValueRanges.SMALL_POSITIVE,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
        )


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        shape_with_kwargs = cls.extend_shape_with_dims_and_keepdims(test_vector.input_shape)
        kwarg_list = []
        for item in shape_with_kwargs:
            kwargs = {}
            kwargs["dim"] = item[1]
            kwargs["keepdim"] = item[2]
            kwarg_list.append(kwargs)
        return kwarg_list

    @classmethod
    def extend_shape_with_dims_and_keepdims(cls, shape):
        shape_with_dims_and_keepdims = list()
        for dim in list(range(0, len(shape), 1)):
            shape_with_dims_and_keepdims.append((shape, dim, True))
            shape_with_dims_and_keepdims.append((shape, dim, False))
        return shape_with_dims_and_keepdims


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "sum",  # 00
            "mean",  # 01
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=TestCollectionCommon.all.input_shapes,
        dev_data_formats=TestCollectionCommon.all.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=TestCollectionCommon.single.dev_data_formats,
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )

    sum = TestCollection(
        operators=[
            "sum",  # 00
        ],
    )

    mean = TestCollection(
        operators=[
            "mean",  # 01
        ],
        dev_data_formats=TestCollectionCommon.float.dev_data_formats,
    )


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test

    failed_mlir_verif_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/errors/test_reduce_ids_failed_mlir_verif_error.txt"
    )

    pcc_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/errors/test_reduce_ids_pcc_error.txt"
    )

    tilize_dtype_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/errors/test_reduce_ids_tilize_dtype_error.txt"
    )

    tilize_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/errors/test_reduce_ids_tilize_error.txt"
    )

    unsupported_dim_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/errors/test_reduce_ids_unsupported_dim_error.txt"
    )


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test plan:
        # 2. Operand source(s):
        # 3. Operand shapes type(s):
        # 4. Operand / output size of dimensions
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test plan:
        # 5. Data format - sum operator
        TestCollection(
            operators=TestCollectionData.sum.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionData.all.dev_data_formats
                if item not in TestCollectionData.single.dev_data_formats
            ],
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # 5. Data format - mean operator
        TestCollection(
            operators=TestCollectionData.mean.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionData.mean.dev_data_formats
                if item not in TestCollectionData.single.dev_data_formats
            ],
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # Test plan:
        # 6. Math fidelity
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[
        # Skip all tests with input shapes with 2 dimensions
        TestCollection(
            criteria=lambda test_vector: len(test_vector.input_shape) == 2,
            skip_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.failed_mlir_verif_error,
            failing_reason=FailingReasons.COMPILATION_FAILED,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.pcc_error,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.tilize_dtype_error,
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.tilize_error,
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.unsupported_dim_error,
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
