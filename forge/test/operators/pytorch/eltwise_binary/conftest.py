# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from test.operators.utils import TestPlanUtils


def pytest_addoption(parser):
    # test id
    parser.addoption("--test_id", action="store", default=None, help="Id of a single op test.")
