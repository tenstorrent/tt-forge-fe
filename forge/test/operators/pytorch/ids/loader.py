# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Loading test ids from files
import os

from typing import Generator, List

from test.operators.utils import TestPlanUtils
from test.operators.utils import TestCollection
from test.operators.utils import FailingReasons
from test.operators.utils import TestSweepsFeatures

from .failing_reasons_register import FailingReasonsRegister


class TestIdsDataLoader:

    __test__ = False  # Avoid collecting TestIdsDataLoader as a pytest test

    @classmethod
    def from_file(cls, test_ids_file: str) -> List[str]:
        return TestPlanUtils.load_test_ids_from_file(f"{os.path.dirname(__file__)}/{test_ids_file}")

    @classmethod
    def _for_failing_reason(cls, operators: List[str], failing_reason: FailingReasons) -> Generator[str, None, None]:
        for operator in operators:
            yield from cls.from_file(f"xfail/{operator}_{failing_reason.name.lower()}.txt")

    @classmethod
    def for_failing_reason(cls, operators: List[str], failing_reason: FailingReasons) -> List[str]:
        return list(cls._for_failing_reason(operators, failing_reason))

    @classmethod
    def build_failing_rule(cls, operators: List[str], failing_reason: FailingReasons) -> TestCollection:
        test_ids = cls.for_failing_reason(operators=operators, failing_reason=failing_reason)
        return TestCollection(
            # operators=operators,  # maybe operators should be explicitly set also in the collection
            criteria=lambda test_vector: test_vector.get_id() in test_ids,
            failing_reason=failing_reason,
        )

    @classmethod
    def build_failing_rules(
        cls, operators: List[str], failing_reasons: List[FailingReasons] = None
    ) -> List[TestCollection]:
        return list(cls._build_failing_rules(operators, failing_reasons))

    @classmethod
    def _build_failing_rules(
        cls, operators: List[str], failing_reasons: List[FailingReasons] = None
    ) -> Generator[TestCollection, None, None]:
        if TestSweepsFeatures.params.ignore_xfail_files:
            return
        for operator in operators:
            if failing_reasons is None:
                op_failing_reasons = list(cls._get_failing_reasons(operator))
            else:
                op_failing_reasons = failing_reasons
            for failing_reason in op_failing_reasons:
                yield cls.build_failing_rule([operator], failing_reason)

    @classmethod
    def _get_failing_reasons(cls, operator: str) -> Generator[FailingReasons, None, None]:
        for failing_reason in FailingReasons:
            for xfail_operator, xfail_reason in FailingReasonsRegister.all:
                if xfail_operator == operator and failing_reason.name == xfail_reason.name:
                    yield failing_reason
