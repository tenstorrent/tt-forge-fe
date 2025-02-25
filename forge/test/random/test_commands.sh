# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Usage:
# source forge/test/random/test_commands.sh
# print_help
# print_params
# test_grapsh


function print_help {
    echo "Help:"
    echo "  print_help          - Print commands and current query parameters."
    echo "  print_query_docs    - Print docs for all available query parameters."
    echo "Query parameters:"
    echo "  print_params        - Print current query parameters values."
    echo "  reset_params        - Reset all query parameters."
    echo "Run commands:"
    echo "  test_grapsh         - Run rgg tests."
    print_params
}

function print_params {
    echo "Query Params:"
    echo "  FRAMEWORKS=$FRAMEWORKS"
    echo "  TEST_NAMES=$TEST_NAMES"
    echo "  RANDOM_TEST_SEED=$RANDOM_TEST_SEED"
    echo "  RANDOM_TEST_COUNT=$RANDOM_TEST_COUNT"
    echo "  VERIFICATION_TIMEOUT=$VERIFICATION_TIMEOUT"
    echo "  MIN_DIM=$MIN_DIM"
    echo "  MAX_DIM=$MAX_DIM"
    echo "  MIN_OP_SIZE_PER_DIM=$MIN_OP_SIZE_PER_DIM"
    echo "  MAX_OP_SIZE_PER_DIM=$MAX_OP_SIZE_PER_DIM"
    echo "  OP_SIZE_QUANTIZATION=$OP_SIZE_QUANTIZATION"
    echo "  MIN_MICROBATCH_SIZE=$MIN_MICROBATCH_SIZE"
    echo "  MAX_MICROBATCH_SIZE=$MAX_MICROBATCH_SIZE"
    echo "  NUM_OF_NODES_MIN=$NUM_OF_NODES_MIN"
    echo "  NUM_OF_NODES_MAX=$NUM_OF_NODES_MAX"
    echo "  NUM_OF_FORK_JOINS_MAX=$NUM_OF_FORK_JOINS_MAX"
    echo "  CONSTANT_INPUT_RATE=$CONSTANT_INPUT_RATE"
    echo "  SAME_INPUTS_PERCENT_LIMIT=$SAME_INPUTS_PERCENT_LIMIT"
    echo "Pytest options:"
    echo "  PYTEST_ADDOPTS = $PYTEST_ADDOPTS"
}

function print_query_docs {
    max_width=$(tput cols)

    pushd ${SCRIPT_DIR}/../../
    python3 -c "from test.random.test_graphs import InfoUtils; InfoUtils.print_query_params(max_width=${max_width})"
    popd
}


function reset_params {
    reset_query_params
    # reset_pytest_opts
}

function reset_query_params {
    unset FRAMEWORKS
    unset TEST_NAMES
    unset RANDOM_TEST_SEED
    unset RANDOM_TEST_COUNT
    unset VERIFICATION_TIMEOUT
    unset MIN_DIM
    unset MAX_DIM
    unset MIN_OP_SIZE_PER_DIM
    unset MAX_OP_SIZE_PER_DIM
    unset OP_SIZE_QUANTIZATION
    unset MIN_MICROBATCH_SIZE
    unset MAX_MICROBATCH_SIZE
    unset NUM_OF_NODES_MIN
    unset NUM_OF_NODES_MAX
    unset NUM_OF_FORK_JOINS_MAX
    unset CONSTANT_INPUT_RATE
    unset SAME_INPUTS_PERCENT_LIMIT
    print_params
}


function _set_pytest_opts {
    local PYTEST_PARAMS=""
    # PYTEST_PARAMS="${PYTEST_PARAMS} -svv"
    PYTEST_PARAMS="${PYTEST_PARAMS} -rap"
    export PYTEST_ADDOPTS="${PYTEST_PARAMS}"
}

function reset_pytest_opts {
    _set_pytest_opts
    print_params
}


function test_graphs {
    print_params
    pytest ${SCRIPT_DIR}/test_graphs.py
    print_params
}


SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

_set_pytest_opts

print_help
# print_query_docs
