cmake_minimum_required(VERSION 3.20.0)
project(ttforge-toolchain LANGUAGES CXX)

# Get parent directory of current source directory
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# Include Utils
list(APPEND CMAKE_MODULE_PATH ${PARENT_DIR}/cmake)
include(Utils)

option(TTFORGE_SKIP_BUILD_TTMLIR_ENV "Skip building the tt-mlir environment" OFF)

# Check if the environment variable is set, if not error out
check_required_env_var(TTFORGE_PYTHON_VERSION)
check_required_env_var(TTFORGE_TOOLCHAIN_DIR)
check_required_env_var(TTFORGE_VENV_DIR)

set(TTFORGE_PYTHON_VERSION $ENV{TTFORGE_PYTHON_VERSION})
set(TTFORGE_TOOLCHAIN_DIR $ENV{TTFORGE_TOOLCHAIN_DIR})
set(TTFORGE_VENV_DIR $ENV{TTFORGE_VENV_DIR})

set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
set(TTMLIR_VENV_DIR $ENV{TTMLIR_VENV_DIR})

if (NOT EXISTS "${TTFORGE_TOOLCHAIN_DIR}")
    message( FATAL_ERROR "The directory ${TTFORGE_TOOLCHAIN_DIR} does not exist. Please create it before running this script.\n  sudo mkdir -p ${TTFORGE_TOOLCHAIN_DIR}\n  sudo chown -R $ENV{USER} ${TTFORGE_TOOLCHAIN_DIR}")
endif()

if (NOT EXISTS "${TTMLIR_TOOLCHAIN_DIR}")
    message( FATAL_ERROR "The directory ${TTMLIR_TOOLCHAIN_DIR} does not exist. Please create it before running this script.\n  sudo mkdir -p ${TTMLIR_TOOLCHAIN_DIR}\n  sudo chown -R $ENV{USER} ${TTMLIR_TOOLCHAIN_DIR}")
endif()

# Install all python dependencies
add_custom_target(python-venv
    ALL COMMAND
    TTFORGE_PYTHON_VERSION=${TTFORGE_PYTHON_VERSION}
    TTFORGE_TOOLCHAIN_DIR=${TTFORGE_TOOLCHAIN_DIR}
    CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    TTFORGE_VENV_DIR=${TTFORGE_VENV_DIR}
    bash ${CMAKE_CURRENT_SOURCE_DIR}/create_venv.sh)

if (NOT TTFORGE_SKIP_BUILD_TTMLIR_ENV)
    add_custom_target(build_tt_mlir_env ALL
        COMMAND ${CMAKE_COMMAND} -E env
        TTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR} # Export TTMLIR_TOOLCHAIN_DIR to use
        TTMLIR_VENV_DIR=${TTMLIR_VENV_DIR} # Export TTMLIR_VENV_DIR to use
        bash ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/build_mlir_env.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tt-mlir
        USES_TERMINAL
    )
endif()
