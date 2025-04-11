# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Loading test ids from files
import os

from typing import Generator, List

from test.operators.utils import TestPlanUtils
from test.operators.utils import TestCollection
from test.operators.utils import FailingReasonsEnum


class TestIdsDataLoader:

    __test__ = False  # Avoid collecting TestIdsDataLoader as a pytest test

    @classmethod
    def from_file(cls, test_ids_file: str) -> List[str]:
        return TestPlanUtils.load_test_ids_from_file(f"{os.path.dirname(__file__)}/{test_ids_file}")

    @classmethod
    def _for_failing_reason(
        cls, operators: List[str], failing_reason: FailingReasonsEnum
    ) -> Generator[str, None, None]:
        for operator in operators:
            yield from cls.from_file(f"xfail/{operator}_{failing_reason.name.lower()}.txt")

    @classmethod
    def for_failing_reason(cls, operators: List[str], failing_reason: FailingReasonsEnum) -> List[str]:
        return list(cls._for_failing_reason(operators, failing_reason))

    @classmethod
    def build_failing_rule(cls, operators: List[str], failing_reason_enum: FailingReasonsEnum) -> TestCollection:
        test_ids = TestIdsDataLoader.for_failing_reason(operators=operators, failing_reason=failing_reason_enum)
        return TestCollection(
            # operators=operators,  # maybe operators should be explicitly set also in the collection
            criteria=lambda test_vector: test_vector.get_id() in test_ids,
            failing_reason=failing_reason_enum.value,
        )

    @classmethod
    def build_failing_rules(
        cls, operators: List[str], failing_reasons: List[FailingReasonsEnum]
    ) -> List[TestCollection]:
        return [cls.build_failing_rule(operators, failing_reason) for failing_reason in failing_reasons]
