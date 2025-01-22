# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Common pytest helper functions


function _set_pytest_opts {
    local PYTEST_PARAMS="${DEFAULT_PYTEST_ADDOPTS}"
    if [ "${_PYTEST_XFAIL_STRICT}" = "false" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} -o xfail_strict=${_PYTEST_XFAIL_STRICT}"
    fi
    if [ "${_PYTEST_NO_SKIPS}" = "true" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} --no-skips"
    fi
    if [ "${_PYTEST_RUN_XFAIL}" = "true" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} --runxfail"  # run xfail tests
    fi
    if [ "${_PYTEST_COLLECT_ONLY}" = "true" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} --collect-only"  # collect only
    fi
    if [ "${_PYTEST_FUNCTION}" != "" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} ${_PYTEST_FUNCTION}"
    fi
    export PYTEST_ADDOPTS="${PYTEST_PARAMS}"
    print_pytest_params
}

function set_default_pytest_opts {
    # By default, xfail_strict is off
    _PYTEST_XFAIL_STRICT=false
}

function reset_pytest_opts {
    unset _PYTEST_XFAIL_STRICT
    unset _PYTEST_RUN_XFAIL
    unset _PYTEST_NO_SKIPS
    unset _PYTEST_COLLECT_ONLY
    set_default_pytest_opts
    _set_pytest_opts
}

function print_pytest_params {
    echo "Pytest options:"
    echo "  PYTEST_ADDOPTS = $PYTEST_ADDOPTS"
}

function reset_test_config_params {
    unset SKIP_FORGE_VERIFICATION
}

function print_test_config_params {
    echo "Test configuration:"
    echo "  SKIP_FORGE_VERIFICATION=$SKIP_FORGE_VERIFICATION"
}

function print_pytest_commands {
    echo "Setup pytest options:"
    echo "  xfail_strict_on     - Enable strict xfail mode via -o xfail_strict=false."
    echo "  xfail_strict_off    - Remove strict xfail mode setup."
    echo "  run_xfail_on        - Enable running xfail tests by including --runxfail."
    echo "  run_xfail_off       - Remove run xfail setup."
    echo "  no_skips_on         - Enable running tests without skips by including --no-skips."
    echo "  no_skips_off        - Remove no skips setup."
    echo "  collect_only_on     - Enable only collecting tests by including --collect-only."
    echo "  collect_only_off    - Remove collect only setup."
    echo "  reset_pytest_opts   - Reset pytest options."
    echo "Setup test configuration:"
    echo "  reset_test_config_params   - Reset test configuration parameters."
}


function xfail_strict_on {
    unset _PYTEST_XFAIL_STRICT
    _set_pytest_opts
}

function xfail_strict_off {
    _PYTEST_XFAIL_STRICT=false
    _set_pytest_opts
}

function run_xfail_on {
    _PYTEST_RUN_XFAIL=true
    _set_pytest_opts
}

function run_xfail_off {
    unset _PYTEST_RUN_XFAIL
    _set_pytest_opts
}

function no_skips_on {
    _PYTEST_NO_SKIPS=true
    _set_pytest_opts
}

function no_skips_off {
    unset _PYTEST_NO_SKIPS
    _set_pytest_opts
}

function collect_only_on {
    _PYTEST_COLLECT_ONLY=true
    _set_pytest_opts
}

function collect_only_off {
    unset _PYTEST_COLLECT_ONLY
    _set_pytest_opts
}

function unset_pytest_function {
    unset _PYTEST_FUNCTION
    _set_pytest_opts
}

function set_pytest_function {
    if [ "${_PYTEST_FUNCTION}" != "$1" ]; then
        _PYTEST_FUNCTION="$1"
        _set_pytest_opts
    fi
}

function with-params {
    print_params
    "$@"
    print_params
}
