# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from test.operators.utils import TestPlanUtils


def pytest_configure(config):
    config.addinivalue_line("markers", 'slow: marks tests as slow (deselect with -m "not slow")')
    config.addinivalue_line("markers", "run_in_pp: marks tests to run in pipeline")


def pytest_addoption(parser):
    # test id
    parser.addoption("--test_id", action="store", default=None, help="Id of a single op test.")


def pytest_generate_tests(metafunc):

    test_id = metafunc.config.option.test_id
    if "single_test_vector" in metafunc.fixturenames and test_id is not None:
        test_plan = TestPlanUtils.build_test_plan_from_id_list([test_id])
        test_vectors = test_plan.generate()
        metafunc.parametrize("single_test_vector", test_vectors)
