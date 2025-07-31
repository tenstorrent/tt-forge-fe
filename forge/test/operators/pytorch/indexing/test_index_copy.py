# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import math

from typing import List, Dict
from loguru import logger

from forge._C import DataFormat

from forge.config import CompilerConfig
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker

from test.operators.utils import VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges
from test.operators.utils.utils import PytorchUtils, TensorUtils
from test.operators.pytorch.ids.loader import TestIdsDataLoader
from test.operators.utils.test_data import TestCollectionTorch


class ModelFromAnotherOp(torch.nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "index_copy_operator_test_op_src_from_another_op"
        self.operator = operator
        self.dim = kwargs.get("dim")
        self.index = kwargs.get("index")

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return self.operator(xx, self.dim, self.index, yy)


class ModelDirect(torch.nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "index_copy_operator_test_op_src_direct"
        self.operator = operator
        self.dim = kwargs.get("dim")
        self.index = kwargs.get("index")

    def forward(self, x, y):
        return self.operator(x, self.dim, self.index, y)
    
class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, self_shape, source_shape, kwargs, dtype, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "index_copy_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.dim = kwargs.get("dim")
        self.index = kwargs.get("index")
        self.c1 = TensorUtils.create_torch_constant(
            input_shape=self_shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=math.prod(self_shape),
        )
        self.c2 = TensorUtils.create_torch_constant(
            input_shape=source_shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=math.prod(source_shape),
        )
        self.register_buffer("constant1", self.c1)
        self.register_buffer("constant2", self.c2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        v1 = self.operator(self.c1, self.dim, self.index, self.c2)
        # v2 = torch.add(x, x)
        v2 = self.operator(x, self.dim, self.index, y)
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
    def make_source_shape(cls, input_shape, dim, index):
        m = len(index)
        source_shape = list(input_shape)
        source_shape[dim] = m
        return tuple(source_shape)

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        input_params: List[Dict] = [],
        warm_reset: bool = False,
    ):
        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)
        
        value_range = ValueRanges.SMALL
        kwargs = test_vector.kwargs if test_vector.kwargs else {}
        
        self_shape = test_vector.input_shape
        source_shape = cls.make_source_shape(test_vector.input_shape, kwargs["dim"], kwargs["index"])
        input_shapes = [self_shape, source_shape]
        logger.trace(f"***input_shapes: {input_shapes}")

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                self_shape=self_shape,
                source_shape=source_shape,
                kwargs=kwargs,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                kwargs=kwargs,
            )

        dtype = kwargs.get("dtype")
        compiler_cfg = CompilerConfig()
        if dtype is torch.bfloat16:
            pytorch_model.to(dtype)
            compiler_cfg.default_df_override = DataFormat.Float16_b

        # Using AllCloseValueChecker, TODO: check if this is appropriate
        verify_config = VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2))
        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            compiler_cfg=compiler_cfg,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            warm_reset=warm_reset,
            value_range=value_range,
            verify_config=verify_config,
            # skip_forge_verification=True,
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None
    
    MAX_INDEX_SIZE = 50

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        max_dim = len(test_vector.input_shape)
        for d in range(max_dim):
            N = test_vector.input_shape[d]
            if N <= 1:
                continue  # no valid index for this dimension
            # m = random.randint(1, N - 1)  # index length
            # m = random.randint(1, 3)  # TODO: should not be limited to 3, should be used line above
            m = random.randint(1, min(N - 1, cls.MAX_INDEX_SIZE))
            index = torch.arange(m, dtype=torch.long)
            yield {
                "index": index,
                "dim": d,
            }

    

class TestCollectionData:

    __test__ = False

    test_collection: TestCollection = None

    operator = ["index_copy"]

# TODO: 
# - add all usual TestCollections: for testing dev_data_format, math_fidelity
# - add case when index tensor is just a single value (doesn't need to be 0)
# - add case when index tensor is a range that starts from non 0 value - this tests are expected to fail
TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection
        TestCollection(
            operators=TestCollectionData.operator,
            # input_sources=[InputSource.FROM_HOST, InputSource.FROM_ANOTHER_OP],
            input_sources=[InputSource.CONST_EVAL_PASS],
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.operator),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]