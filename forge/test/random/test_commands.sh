# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Usage:
# source forge/test/random/test_commands.sh


function print_help {
    echo "Help:"
    echo "  print_help              - Print commands and current query parameters."
    echo "  print_query_docs        - Print docs for all available query parameters."
    echo "  print_pytest_commands   - Print pytest helper commands."
    echo "Query parameters:"
    echo "  print_params            - Print current query parameters values."
    echo "  reset_params            - Reset all query and test configuration parameters."
    echo "Setup commands:"
    echo "  select_test_graphs      - Select test_graphs pytest."
    echo "Example commands:"
    echo "  RANDOM_TEST_COUNT=5 FRAMEWORKS=PYTORCH with-params pytest"
    echo "Run commands:"
    echo "  pytest                  - Run all tests specfied via a query parameters."
    echo "  with-params pytest      - Print params before and after test run."
    print_params
}

function print_params {
    echo "Query Params:"
    echo "  FRAMEWORKS=$FRAMEWORKS"
    echo "  ALGORITHMS=$ALGORITHMS"
    echo "  CONFIGS=$CONFIGS"
    echo "  RANDOM_TEST_SEED=$RANDOM_TEST_SEED"
    echo "  RANDOM_TEST_COUNT=$RANDOM_TEST_COUNT"
    echo "  RANDOM_TESTS_SELECTED=$RANDOM_TESTS_SELECTED"
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

    pushd ${RGG_SCRIPT_DIR}/../../
    python3 -c "from test.random.test_graphs import InfoUtils; InfoUtils.print_query_params(max_width=${max_width})"
    popd
}


function reset_query_params {
    unset FRAMEWORKS
    unset ALGORITHMS
    unset CONFIGS
    unset RANDOM_TEST_SEED
    unset RANDOM_TEST_COUNT
    unset RANDOM_TESTS_SELECTED
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


function filters_to_string {
    filters=""
    filters=""
    if [ "${FRAMEWORKS}" != "" ]; then
        filters="${filters}_FRAMEWORKS_${FRAMEWORKS}"
    fi
    if [ "${ALGORITHMS}" != "" ]; then
        filters="${filters}ALGORITHMS${ALGORITHMS}"
    fi
    if [ "${CONFIGS}" != "" ]; then
        filters="${filters}_CONFIGS_${CONFIGS}"
    fi
    if [ "${RANDOM_TEST_SEED}" != "" ]; then
        filters="${filters}_SEED_${RANDOM_TEST_SEED}"
    fi
    if [ "${RANDOM_TEST_COUNT}" != "" ]; then
        filters="${filters}_COUNT_${RANDOM_TEST_COUNT}"
    fi
    if [[ "${NUM_OF_NODES_MIN}" != "" || "${NUM_OF_NODES_MAX}" != "" ]]; then
        filters="${filters}_NUM_OF_NODES_${NUM_OF_NODES_MIN}_${NUM_OF_NODES_MAX}"
    fi
    filters=$(
        echo "${filters}" \
        | sed 's/[ {}'"'"']//g' \
        | sed 's/[ ,:\(\)\/]/_/g' \
    )
    echo "${filters}"
}


function select_test_graphs {
    set_pytest_function "${RGG_SCRIPT_DIR}/test_graphs.py"
    # set_pytest_function "forge/test/random/test_graphs.py"
}


RGG_SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

. ${RGG_SCRIPT_DIR}/../operators/pytorch/test_common.sh

set_default_pytest_opts
select_test_graphs
# _set_pytest_opts

print_help
