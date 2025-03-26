# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of embedding operators
#
# In this test we test pytorch embedding operator

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


from functools import reduce
import math
import random
import pytest

from typing import List, Dict, Type, Optional, Any

import torch
import forge
import forge.op
from loguru import logger


from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils.datatypes import ValueRange


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "num_embeddings": kwargs["num_embeddings"],
            "embedding_dim": kwargs["embedding_dim"],
        }

        self.weight = torch.rand(
            (self.kwargs["num_embeddings"], self.kwargs["embedding_dim"]), dtype=kwargs["weight_dtype"]
        )
        self.embedding = self.operator(**self.kwargs, _weight=self.weight)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the embedding operator
        add = torch.add(x, x)
        output = self.embedding(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "num_embeddings": kwargs["num_embeddings"],
            "embedding_dim": kwargs["embedding_dim"],
        }

        self.weight = torch.rand(
            (self.kwargs["num_embeddings"], self.kwargs["embedding_dim"]), dtype=kwargs["weight_dtype"]
        )
        self.embedding = self.operator(**self.kwargs, _weight=self.weight)

    def forward(self, x: torch.Tensor):
        output = self.embedding(x)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "num_embeddings": kwargs["num_embeddings"],
            "embedding_dim": kwargs["embedding_dim"],
        }

        self.constant = torch.randint(0, self.kwargs["num_embeddings"] - 1, self.shape, dtype=torch.int32)

        self.weight = torch.rand(
            (self.kwargs["num_embeddings"], self.kwargs["embedding_dim"]), dtype=kwargs["weight_dtype"]
        )
        self.embedding = self.operator(**self.kwargs, _weight=self.weight)

    def forward(self, x: torch.Tensor):
        v1 = self.embedding(self.constant)
        # v2 = torch.add(x, x)
        v2 = self.embedding(x)
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
        input_params: List[Dict] = [],
        number_of_operands: int = 1,
        warm_reset: bool = False,
    ):
        """Common verification function for all tests"""

        operator = getattr(torch.nn, test_vector.operator)

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

        min = 0
        max = test_vector.kwargs["num_embeddings"] - 1
        match test_vector.input_source:
            case InputSource.FROM_ANOTHER_OP:
                max = int(max / 2)
        value_range = ValueRange(min, max)

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
            value_range=value_range,
        )


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    # rng = random.Random(31)
    INPUT_SHAPE_THRESHOLD = 100000000
    MAX_EMBEDDING_DIM = 10000

    embedding_dims = [1000, 3200, MAX_EMBEDDING_DIM]
    # weight_dtypes = [torch.bfloat16, torch.float32]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector, weight_dtype: Type[torch.dtype]):
        num_embedding_limit = 2**7 - 1  # 127
        rng = random.Random(math.prod(test_vector.input_shape) + 1)
        num_embeddings = []
        match test_vector.dev_data_format:
            case forge.DataFormat.RawUInt8:
                num_embedding_limit = 2**8 - 1  # 255
                num_embeddings = [rng.randint(2, num_embedding_limit)]
            case forge.DataFormat.RawUInt16:
                num_embedding_limit = 2**16 - 1  # 65535
                num_embeddings = [rng.randint(2, num_embedding_limit)]
            case forge.DataFormat.RawUInt32:
                num_embedding_limit = 2**32 - 1  # 4294967295
                num_embeddings = [rng.randint(2, 32000)]
            case forge.DataFormat.Int8:
                num_embedding_limit = 2**7 - 1  # 127
                num_embeddings = [rng.randint(2, num_embedding_limit)]
            case forge.DataFormat.UInt16:
                num_embedding_limit = 2**16 - 1  # 65535
                num_embeddings = [rng.randint(2, num_embedding_limit)]
            case forge.DataFormat.Int32:
                num_embedding_limit = 2**31 - 1  # 2147483647
                num_embeddings = [rng.randint(2, 32000)]

        kwarg_list = []
        for num_embeddings in num_embeddings:
            for embedding_dim in cls.embedding_dims:
                kwarg_list.append(
                    {
                        "num_embeddings": num_embeddings,
                        "embedding_dim": embedding_dim,
                        "weight_dtype": weight_dtype,
                    }
                )
        return kwarg_list


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "Embedding",  # 00
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=[
            input_shape
            for input_shape in TestCollectionCommon.all.input_shapes
            if reduce(lambda x, y: x * y, input_shape) * TestParamsData.MAX_EMBEDDING_DIM
            < TestParamsData.INPUT_SHAPE_THRESHOLD
        ],
        dev_data_formats=TestCollectionCommon.int.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=[
            pytest.param(forge.DataFormat.Int32, id="Int32"),
        ],
        math_fidelities=TestCollectionCommon.single.math_fidelities,
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
            input_shapes=TestCollectionData.single.input_shapes,
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector, torch.float32),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            dev_data_formats=TestCollectionData.all.dev_data_formats,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector, torch.bfloat16),
        ),
        # Test plan:
        # 6. Math fidelity
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector, torch.bfloat16),
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[
        # FLOAT32 ERRORS:
        # RuntimeError: Fatal error
        # FATAL | Input output tensor size mismatch in memcpy: 40000 * 4 != 240000 * 4
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
                InputSource.FROM_ANOTHER_OP,
            ],
            criteria=lambda test_vector: test_vector.kwargs["weight_dtype"] == torch.float32,
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        # AssertionError: PCC check failed
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            criteria=lambda test_vector: test_vector.kwargs["weight_dtype"] == torch.float32,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # BFLOAT16 ERRORS:
        # PCC ERRORS:
        # AssertionError: PCC check failed
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
            ],
            input_shapes=[
                (45, 17),
                (9920, 1),
                (17, 41),
                (89, 3),
                (11, 1, 23),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            input_sources=[
                InputSource.FROM_ANOTHER_OP,
            ],
            criteria=lambda test_vector: len(test_vector.input_shape) == 2
            or test_vector.input_shape in ((1, 1, 23), (11, 1, 23)),
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # RUNTIME ERRORS:
        # RuntimeError: Tensor 1 - data type mismatch: expected BFloat16, got Float32
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            criteria=lambda test_vector: test_vector.kwargs["weight_dtype"] == torch.bfloat16,
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        # ALLOCATION ERRORS:
        # RuntimeError: TT_THROW @ /home/kmilanovic/src/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/allocator.cpp:145: tt::exception
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
                InputSource.FROM_ANOTHER_OP,
            ],
            input_shapes=[
                (9920, 1),
            ],
            kwargs=[
                {
                    "embedding_dim": 10000,
                },
            ],
            failing_reason=FailingReasons.ALLOCATION_FAILED,
        ),
        # FATAL ERRORS:
        # RuntimeError: Fatal error
        # FATAL | Input output tensor size mismatch in memcpy: 40000 * 4 != 240000 * 4
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
            ],
            criteria=lambda test_vector: len(test_vector.input_shape) > 2
            and test_vector.input_shape not in ((1, 1, 23), (11, 1, 23)),
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        TestCollection(
            input_sources=[
                InputSource.FROM_ANOTHER_OP,
            ],
            criteria=lambda test_vector: len(test_vector.input_shape) > 2
            and test_vector.input_shape not in ((1, 1, 23), (11, 1, 23)),
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        # SEGMENTATION FAULT ERROR
        TestCollection(
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=[
                (9920, 1),
                (1, 9920, 1),
                (1, 1, 9920, 1),
            ],
            kwargs=[
                {
                    "embedding_dim": 10000,
                },
            ],
            skip_reason=FailingReasons.SEG_FAULT,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
