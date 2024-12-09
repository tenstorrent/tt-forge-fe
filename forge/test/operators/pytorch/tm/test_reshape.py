# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# TODO: Add test plan header here

import torch
import random
import os

from typing import List, Dict
from loguru import logger

from forge.verify.config import VerifyConfig

from forge.verify.value_checkers import AllCloseValueChecker

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import TestPlanUtils
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges

from test.operators.pytorch.eltwise_unary import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


def generate_target_shape(source_shape, max_dims=4, seed=4):
    """Generates a meaningful target shape based on the source_shape with a limit on the number of dimensions.
    Ensures that the target shape is not the same as the source shape and that the number of dimensions
    does not exceed the max_dims parameter."""

    rng = random.Random(seed)

    total_elements = torch.prod(torch.tensor(source_shape)).item()
    num_attempts = 0  # To prevent infinite loops

    while True:
        num_dims = rng.randint(1, max_dims)  # Limit the maximum number of dimensions

        # Attempt to divide total_elements into a random number of factors
        target_shape = []
        remaining = total_elements
        for i in range(num_dims - 1):
            if remaining == 1:
                target_shape.extend([1] * (num_dims - len(target_shape)))
                break
            dim_size = rng.randint(1, remaining)
            while remaining % dim_size != 0:  # Ensure that dim_size divides remaining
                dim_size = rng.randint(1, remaining)
            target_shape.append(dim_size)
            remaining //= dim_size

        target_shape.append(remaining)  # Add the final dimension
        rng.shuffle(target_shape)  # Shuffle the dimensions for additional randomness

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
            warm_reset=warm_reset,
            value_range=ValueRanges.SMALL,
            deprecated_verification=False,
            verify_config=VerifyConfig(value_checker=AllCloseValueChecker()),
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    # fmt: off
    specific_reshapes = {
        # Flatten Reshapes:
        (8, 8, 8):        [(512,)],
        (1, 2, 2, 2):     [(4, 2), (8,), ],
        # Dynamic Shape Reshapes:
        (1, 49, 2304):    [(1, -1, 3, 24, 32),
                           (1, 49, 6, -1),
                           (-1, ),
                           (1, -1)],
        (3, 4, 5):        [(3, -1), (-1, 15)],
        (2, 2, 2, 2):     [(4, -1), (8, -1)],
        # Collapse Rank Reshapes:
        (2, 3, 4, 5):     [(6, 4, 5), (4, 6, 5)],
        (2, 2, 3, 4):     [(4, 3, 4), (2, 6, 4)],
        (3, 3, 3, 3):     [(9, 3, 3), (3, 9, 3)],
        (1, 2, 3, 4, 5):  [(2, 3, 4, 5), (6, 4, 5)],
        (2, 2, 2, 2, 2):  [(4, 2, 2, 2), (8, 2, 2)],
        (2, 3, 4, 8):     [(6, 4, 8), (24, 8)],
        # Expand Rank Reshapes:
        (6, 4, 5):        [(2, 3, 4, 5),
                           (1, 6, 4, 5),
                           (2, 3, 2, 2, 5)],
        (12, 8, 10):      [(2, 6, 8, 10),
                           (2, 2, 3, 8, 10),
                           (1, 3, 4, 8, 10)],
        # Channel Manipulation Reshapes:
        (1, 32, 2560):    [(1, 32, 20, 128),
                           (1, 32, 40, 64),
                           (1, 32, 10, 256),
                           (1, 32, 5, 512),
                           (1, 32, 80, 32)],
        (1, 64, 1280):    [(1, 64, 10, 128),
                           (1, 64, 20, 64),
                           (1, 64, 5, 256),
                           (1, 64, 40, 32),
                           (1, 64, 80, 16)],
        # Edge Case Reshapes:
        (0,):             [(0, 0, 1),
                           (0, 1, 0),
                           (1, 0, 0),
                           (0, 0, 10)],
    }
    # fmt: on

    @classmethod
    def generate_random_kwargs(cls, test_vector: TestVector):
        """
        This method generates a random target shape based on the input shape of the
        test vector and a seed value derived from the sum of the elements in the
        input shape. The generated target shape is then returned as a dictionary
        within a list.
        Args:
            test_vector (TestVector): The test vector containing the input shape.
        Returns:
            list: A list containing a dictionary with the generated target shape.
        """

        input_shape = test_vector.input_shape
        seed = 0
        for i in range(len(input_shape)):
            seed += input_shape[i]
        target_shape = generate_target_shape(test_vector.input_shape, seed=seed)
        return [
            {"shape": target_shape},
        ]

    @classmethod
    def generate_specific_kwargs(cls, test_vector: TestVector):
        """
        Generate specific keyword arguments for reshaping tensors based on the given test vector.
        Args:
            cls: The class that contains specific reshapes mapping.
            test_vector (TestVector): An instance of TestVector containing the input shape.
        Returns:
            list: A list of dictionaries, each containing a target shape for reshaping.
        """

        input_shape = test_vector.input_shape
        target_shapes = []
        for item in cls.specific_reshapes[input_shape]:
            target_shapes.append({"shape": item})
        return target_shapes


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
            kwargs=lambda test_vector: TestParamsData.generate_random_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_random_kwargs(test_vector),
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
            kwargs=lambda test_vector: TestParamsData.generate_random_kwargs(test_vector),
            dev_data_formats=TestCollectionCommon.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
        # Test specific classes of reshape operations collection:
        TestCollection(
            operators=["reshape"],
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestParamsData.specific_reshapes.keys(),
            kwargs=lambda test_vector: TestParamsData.generate_specific_kwargs(test_vector),
        ),
    ],
    failing_rules=[
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id()
            in TestPlanUtils.load_test_ids_from_file(
                f"{os.path.dirname(__file__)}/test_reshape_ids_failed_allclose_value_checker.txt"
            ),
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            input_shapes=[(1, 10000)],
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        TestCollection(
            input_shapes=[
                (100, 100),
                (1000, 100),
                (89, 3),
                (1, 64, 1),
                (1, 100, 100),
                (11, 17, 41),
                (1, 2, 3, 4),
                (1, 11, 45, 17),
                (1, 11, 17, 41),
                (1, 13, 89, 3),
                (8, 1, 10, 1000),
                (11, 32, 32, 64),
                (8, 8, 8),
            ],
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        TestCollection(
            input_shapes=[(1, 2, 2, 2)],
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "shape" in test_vector.kwargs
            and test_vector.kwargs["shape"] == (8,),
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        TestCollection(
            input_shapes=[(1, 49, 2304)],
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "shape" in test_vector.kwargs
            and test_vector.kwargs["shape"] == (-1,),
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
        TestCollection(
            input_shapes=[(0,)],
            failing_reason=FailingReasons.UNSUPPORTED_DIMENSION,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
