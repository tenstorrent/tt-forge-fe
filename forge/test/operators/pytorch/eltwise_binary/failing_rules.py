# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing rules for element-wise binary operators


import forge

from test.operators.utils import InputSource
from test.operators.utils import TestCollection
from test.operators.utils import TestResultFailing
from test.operators.utils import FailingReasons
from test.operators.utils import TestCollectionCommon


class FailingRulesData:

    common = [
        # PCC check fails for all input sources and buggy shapes
        TestCollection(
            input_shapes=[
                (1, 1),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for CONST_EVAL_PASS and buggy shapes
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=[
                (11, 45, 17),
                (10, 1000, 100),
                (10, 10000, 1),
                (32, 32, 64),
                # fail only for const eval pass not for other models
                (2, 3, 4),
                (11, 1, 23),
                (11, 64, 1),
                (100, 100, 100),
                (64, 160, 96),
                (11, 17, 41),
                (13, 89, 3),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for FROM_HOST and all int dev data formats
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
            ],
            dev_data_formats=TestCollectionCommon.int.dev_data_formats,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
    ]

    add = [
        # PCC check fails for all input sources and buggy shapes
        common[0],
        # PCC check fails for CONST_EVAL_PASS and buggy shapes
        common[1],
        # PCC check fails for FROM_HOST and all int dev data formats
        common[2],
        # PCC check fails for buggy shapes for add
        TestCollection(
            input_sources=[
                InputSource.FROM_DRAM_QUEUE,
                InputSource.CONST_EVAL_PASS,
            ],
            dev_data_formats=[
                None,
            ],
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "alpha" in test_vector.kwargs
            and test_vector.kwargs["alpha"] != 1,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            dev_data_formats=[
                None,
            ],
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "alpha" in test_vector.kwargs
            and len(test_vector.input_shape) == 2
            and test_vector.input_shape != (1, 1)
            and test_vector.input_shape[-1] == 1,
            failing_reason=None,
        ),
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=[
                (1, 3),
            ],
            kwargs=[
                {
                    "alpha": 0.17234435,
                },
            ],
            # dev_data_formats=[None,],
            failing_reason=None,
        ),
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
            ],
            # dev_data_formats=TestCollectionCommon.float.dev_data_formats,
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "alpha" in test_vector.kwargs
            and test_vector.kwargs["alpha"] != 1,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            input_sources=[
                InputSource.FROM_ANOTHER_OP,
            ],
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "alpha" in test_vector.kwargs
            and test_vector.kwargs["alpha"] < 0,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
    ]

    sub = [
        # PCC check fails for all input sources and buggy shapes
        common[0],
        # Exception from DATA_MISMATCH
        TestCollection(
            input_sources=[
                InputSource.FROM_ANOTHER_OP,
            ],
            input_shapes=[
                (1, 1),
            ],
            failing_reason=None,
        ),
        # PCC check fails for CONST_EVAL_PASS and buggy shapes
        common[1],
        # PCC check fails for FROM_HOST and all int dev data formats
        common[2],
        TestCollection(
            input_sources=[
                InputSource.FROM_ANOTHER_OP,
            ],
            input_shapes=[
                (1, 1),
            ],
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "alpha" in test_vector.kwargs
            and test_vector.kwargs["alpha"] != 1,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
    ]

    mul = [
        # PCC check fails for all input sources and buggy shapes
        common[0],
        # PCC check fails for CONST_EVAL_PASS and buggy shapes
        common[1],
        # PCC check fails for FROM_HOST and all int dev data formats
        common[2],
        # Exception from DATA_MISMATCH
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=[
                (2, 3, 4),
            ],
            failing_reason=None,
        ),
    ]

    div = [
        # PCC check fails for all input sources and buggy shapes
        common[0],
        # PCC check fails for CONST_EVAL_PASS and buggy shapes
        common[1],
        # PCC check fails for FROM_HOST and all int dev data formats
        common[2],
        # Failing when testing with LARGE
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
            ],
            # input_shapes=[
            #     (1, 2, 3, 4),
            # ],
            dev_data_formats=TestCollectionCommon.float.dev_data_formats,
            kwargs=[
                {
                    "rounding_mode": "trunc",
                },
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for div
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
                InputSource.FROM_DRAM_QUEUE,
            ],
            input_shapes=[
                (1, 4),  # TODO remove fixed
                (1, 3),  # TODO remove fixed
                (3, 4),  # TODO remove fixed
                (1, 3, 4),  # TODO remove fixed
                # (12, 64, 160, 96),
            ],
            kwargs=[
                {
                    "rounding_mode": "trunc",
                },
                {
                    "rounding_mode": "floor",
                },
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for div
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=[
                (1, 17),  # TODO remove fixed
                (45, 17),  # TODO remove fixed
            ],
            kwargs=[
                {
                    "rounding_mode": "trunc",
                },
                {
                    "rounding_mode": "floor",
                },
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for div
        TestCollection(
            input_sources=[
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=[
                (1, 41),
                (17, 41),
                (1, 2, 3, 4),
                (2, 2, 3, 4),
            ],
            kwargs=[
                {
                    "rounding_mode": "trunc",
                },
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
    ]

    ge = [
        # PCC check fails for all input sources and buggy shapes
        common[0],
        # Exception from DATA_MISMATCH
        TestCollection(
            input_sources=[
                InputSource.FROM_ANOTHER_OP,
                InputSource.FROM_HOST,
                InputSource.FROM_DRAM_QUEUE,
            ],
            input_shapes=[
                (1, 1),
            ],
            failing_reason=None,
        ),
        # PCC check fails for CONST_EVAL_PASS and buggy shapes
        common[1],
        # PCC check fails for FROM_HOST and all int dev data formats
        common[2],
        # PCC check fails for buggy shapes for ge
        TestCollection(
            input_sources=[
                InputSource.FROM_HOST,
                InputSource.FROM_DRAM_QUEUE,
            ],
            input_shapes=[
                (1, 1000),
                (5, 11, 64, 1),
                # fail when dtype=float32 or generator
                # (17, 41),
                # (89, 3),
                # (1, 17, 41),
                # (1, 89, 3),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
    ]
