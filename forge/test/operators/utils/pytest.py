# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# pytest utilities

import _pytest
import _pytest.reports


class PyTestUtils:

    @classmethod
    def get_xfail_reason(cls, item: _pytest.python.Function) -> str:
        ''' Get xfail reason from pytest item
        
        Args:
            item (_pytest.python.Function): Pytest item

        Returns:
            str: Xfail reason
        '''
        xfail_marker = item.get_closest_marker("xfail")

        if xfail_marker:
            xfail_reason = xfail_marker.kwargs.get("reason", "No reason provided")
            return xfail_reason
    
        return None
