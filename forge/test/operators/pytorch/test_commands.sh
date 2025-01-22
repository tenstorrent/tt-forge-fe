# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Usage:
# source forge/test/operators/pytorch/test_commands.sh


function print_help {
    echo "Help:"
    echo "  print_help              - Print commands and current query parameters."
    echo "  print_query_docs        - Print docs for all available query parameters."
    echo "  print_pytest_commands   - Print pytest helper commands."
    echo "Query parameters:"
    echo "  print_params            - Print current query parameters values."
    echo "  reset_params            - Reset all query and test configuration parameters."
    echo "Setup commands:"
    echo "  select_test_query       - Select test_query pytest."
    echo "Example commands:"
    echo "  OPERATORS=add with-params pytest"
    echo "Run commands:"
    echo "  pytest                  - Run all tests or subset of test plan based on a query parameters."
    echo "  with-params pytest      - Print params before and after test run."
    print_params
}

function print_params {
    echo "Query Params:"
    echo "  OPERATORS=$OPERATORS"
    echo "  FILTERS=$FILTERS"
    echo "  INPUT_SOURCES=$INPUT_SOURCES"
    echo "  INPUT_SHAPES=\"$INPUT_SHAPES\""
    echo "  DEV_DATA_FORMATS=$DEV_DATA_FORMATS"
    echo "  MATH_FIDELITIES=$MATH_FIDELITIES"
    echo "  KWARGS=\"$KWARGS\""
    echo "  FAILING_REASONS=$FAILING_REASONS"
    echo "  SKIP_REASONS=$SKIP_REASONS"
    echo "  RANDOM_SEED=$RANDOM_SEED"
    echo "  SAMPLE=$SAMPLE"
    echo "  UNIQUE_KWARGS=$UNIQUE_KWARGS"
    echo "  RANGE=$RANGE"
    echo "  TEST_ID=$TEST_ID"
    echo "  ID_FILES=$ID_FILES"
    echo "  ID_FILES_IGNORE=$ID_FILES_IGNORE"
    print_test_config_params
    print_pytest_params
}

function reset_params {
    reset_query_params
    reset_test_config_params
    # reset_pytest_opts
}

function print_query_docs {
    max_width=$(tput cols)
    max_width=$((max_width * 80 / 100))

    pushd ${SCRIPT_DIR}/../../../
    python3 -c "from test.operators.pytorch.test_all import InfoUtils; InfoUtils.print_query_params(max_width=${max_width})"
    popd
}


function reset_query_params {
    unset OPERATORS
    unset FILTERS
    unset INPUT_SOURCES
    unset INPUT_SHAPES
    unset DEV_DATA_FORMATS
    unset MATH_FIDELITIES
    unset KWARGS
    unset FAILING_REASONS
    unset SKIP_REASONS
    unset RANDOM_SEED
    unset SAMPLE
    unset UNIQUE_KWARGS
    unset RANGE
    unset TEST_ID
    unset ID_FILES
    unset ID_FILES_IGNORE
    print_params
}


function select_test_query {
    set_pytest_function "${SCRIPT_DIR}/test_all.py::test_query"
    # set_pytest_function "forge/test/operators/pytorch/test_all.py::test_query"
}


SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

. ${SCRIPT_DIR}/test_common.sh

set_default_pytest_opts
select_test_query
# _set_pytest_opts

print_help
