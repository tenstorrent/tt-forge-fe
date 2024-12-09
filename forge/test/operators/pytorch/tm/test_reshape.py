# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# TODO: Add test plan header here

import torch
import random

from typing import List, Dict
from loguru import logger
from forge import MathFidelity, DataFormat

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges

from test.operators.pytorch.eltwise_unary import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


def generate_target_shape(source_shape, max_dims=4, seed=None):
    """Generates a meaningful target shape based on the source_shape with a limit on the number of dimensions.
       Ensures that the target shape is not the same as the source shape and that the number of dimensions 
       does not exceed the max_dims parameter."""
    
    if seed is not None:
        random.seed(seed)  # Set the random seed to ensure reproducibility
    
    total_elements = torch.prod(torch.tensor(source_shape)).item()
    num_attempts = 0  # To prevent infinite loops
    
    while True:
        num_dims = random.randint(1, max_dims)  # Limit the maximum number of dimensions
        
        # Attempt to divide total_elements into a random number of factors
        target_shape = []
        remaining = total_elements
        for i in range(num_dims - 1):
            if remaining == 1:
                target_shape.extend([1] * (num_dims - len(target_shape)))
                break
            dim_size = random.randint(1, remaining)
            while remaining % dim_size != 0:  # Ensure that dim_size divides remaining
                dim_size = random.randint(1, remaining)
            target_shape.append(dim_size)
            remaining //= dim_size
        
        target_shape.append(remaining)  # Add the final dimension
        random.shuffle(target_shape)  # Shuffle the dimensions for additional randomness
        
        if tuple(target_shape) != source_shape and len(target_shape) <= max_dims:  # Ensure target shape is valid
            break
        
        num_attempts += 1
        if num_attempts > 100:  # Prevent infinite loop if no valid shape can be found
            raise ValueError(f"Failed to generate a target shape different from {source_shape} after 100 attempts.")
    
    return tuple(target_shape)


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.FROM_DRAM_QUEUE: ModelDirect,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        input_params: List[Dict] = [],
        warm_reset: bool = False,
    ):

        input_source_flag: InputSourceFlags = None
        if test_vector.input_source in (InputSource.FROM_DRAM_QUEUE,):
            input_source_flag = InputSourceFlags.FROM_DRAM

        operator = getattr(torch, test_vector.operator)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(operator, test_vector.input_shape, kwargs)
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator, kwargs)
        )

        input_shapes = tuple([test_vector.input_shape])

        logger.trace(f"***input_shapes: {input_shapes}")

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            input_source_flag=input_source_flag,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
            value_range=ValueRanges.SMALL,
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        target_shape = generate_target_shape(test_vector.input_shape, seed=len(test_vector.input_shape))
        return [
            {"shape": target_shape},
        ]


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionCommon.all.dev_data_formats
                if item not in TestCollectionCommon.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test Math fidelities collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionCommon.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        # Skip 2D shapes as we don't test them:
        TestCollection(
            criteria=lambda test_vector: len(test_vector.input_shape) in (2,),
            skip_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
