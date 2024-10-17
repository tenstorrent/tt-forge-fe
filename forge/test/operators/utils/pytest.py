# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# pytest utilities

import _pytest
import _pytest.reports

from _pytest.mark import ParameterSet


class PyTestUtils:
    @classmethod
    def get_xfail_reason(cls, item: _pytest.python.Function) -> str:
        """Get xfail reason from pytest item

        Args:
            item (_pytest.python.Function): Pytest item

        Returns:
            str: Xfail reason
        """
        xfail_marker = item.get_closest_marker("xfail")

        if xfail_marker:
            xfail_reason = xfail_marker.kwargs.get("reason", "No reason provided")
            return xfail_reason

        return None


class PytestParamsUtils:
    @classmethod
    def strip_param_set(cls, value):
        if isinstance(value, ParameterSet):
            value = value[0][0]
        return value

    @classmethod
    def strip_param_sets(cls, values):
        return [cls.strip_param_set(value) for value in values]
