# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Usage:
# source forge/test/operators/pytorch/test_commands.sh
# print_help
# collect_only_on
# test_plan
# test_query
# test_unique


function print_help {
    echo "Help:"
    echo "  print_help          - Print commands and current query parameters."
    echo "  print_query_docs    - Print docs for all available query parameters."
    echo "Query parameters:"
    echo "  print_params        - Print current query parameters values."
    echo "  reset_params        - Reset all query parameters and pytest options."
    echo "  reset_query_params  - Reset all query parameters."
    echo "Setup pytest options:"
    echo "  run_xfail_on        - Enable running xfail tests by including --runxfail."
    echo "  run_xfail_off       - Remove run xfail setup."
    echo "  no_skips_on         - Enable running tests without skips by including --no-skips."
    echo "  no_skips_off        - Remove no skips setup."
    echo "  collect_only_on     - Enable only collecting tests by including --collect-only."
    echo "  collect_only_off    - Remove collect only setup."
    echo "  reset_pytest_opts   - Reset pytest options."
    echo "Run commands:"
    echo "  test_plan           - Run all tests from test plan. Support OPERATORS query parameter."
    echo "  test_query          - Run subset of test plan based on a query parameters."
    echo "  test_unique         - Run representative examples of all available tests."
    echo "  test_single         - Run single test based on TEST_ID parameter."
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
    echo "  RANGE=$RANGE"
    echo "  TEST_ID=$TEST_ID"
    echo "Pytest options:"
    echo "  PYTEST_ADDOPTS = $PYTEST_ADDOPTS"
}

function reset_params {
    reset_query_params
    reset_pytest_opts
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
    unset RANGE
    unset TEST_ID
    print_params
}


function run_xfail_on {
    _PYTEST_RUN_XFAIL=true
    _set_pytest_opts
    print_params
}

function run_xfail_off {
    unset _PYTEST_RUN_XFAIL
    _set_pytest_opts
    print_params
}

function no_skips_on {
    _PYTEST_NO_SKIPS=true
    _set_pytest_opts
    print_params
}

function no_skips_off {
    unset _PYTEST_NO_SKIPS
    _set_pytest_opts
    print_params
}

function collect_only_on {
    _PYTEST_COLLECT_ONLY=true
    _set_pytest_opts
    print_params
}

function collect_only_off {
    unset _PYTEST_COLLECT_ONLY
    _set_pytest_opts
    print_params
}

function _set_pytest_opts {
    local PYTEST_PARAMS=""
    # PYTEST_PARAMS="${PYTEST_PARAMS} -svv"
    PYTEST_PARAMS="${PYTEST_PARAMS} -rap"
    if [ "${_PYTEST_NO_SKIPS}" = "true" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} --no-skips"
    fi
    if [ "${_PYTEST_RUN_XFAIL}" = "true" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} --runxfail"  # run xfail tests
    fi
    if [ "${_PYTEST_COLLECT_ONLY}" = "true" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} --collect-only"  # collect only
    fi
    export PYTEST_ADDOPTS="${PYTEST_PARAMS}"
}

function reset_pytest_opts {
    unset _PYTEST_RUN_XFAIL
    unset _PYTEST_NO_SKIPS
    unset _PYTEST_COLLECT_ONLY
    _set_pytest_opts
    print_params
}


function _run_test_all {
    function_name=$1
    print_params
    pytest ${SCRIPT_DIR}/test_all.py::${function_name}
    print_params
}

function test_plan {
    _run_test_all test_plan
}

function test_query {
    _run_test_all test_query
}

function test_unique {
    _run_test_all test_unique
}

function test_single {
    _run_test_all test_single
}


SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
# echo "SCRIPT_DIR = ${SCRIPT_DIR}"

_set_pytest_opts

print_help
# print_query_docs
