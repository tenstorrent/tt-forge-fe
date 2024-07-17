cmake_minimum_required(VERSION 3.20.0)
project(pybuda-toolchain LANGUAGES CXX)

# Get parent directory of current source directory
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# Include Utils
list(APPEND CMAKE_MODULE_PATH ${PARENT_DIR}/cmake)
include(Utils)

# Check if the environment variable is set, if not error out
check_required_env_var(PYBUDA_PYTHON_VERSION)
check_required_env_var(PYBUDA_TOOLCHAIN_DIR)
check_required_env_var(PYBUDA_VENV_DIR)

set(PYBUDA_PYTHON_VERSION $ENV{PYBUDA_PYTHON_VERSION})
set(PYBUDA_TOOLCHAIN_DIR $ENV{PYBUDA_TOOLCHAIN_DIR})
set(PYBUDA_VENV_DIR $ENV{PYBUDA_VENV_DIR})

set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
set(TTMLIR_VENV_DIR $ENV{TTMLIR_VENV_DIR})

if (NOT EXISTS "${PYBUDA_TOOLCHAIN_DIR}")
    message( FATAL_ERROR "The directory ${PYBUDA_TOOLCHAIN_DIR} does not exist. Please create it before running this script.\n  sudo mkdir -p ${PYBUDA_TOOLCHAIN_DIR}\n  sudo chown -R $ENV{USER} ${PYBUDA_TOOLCHAIN_DIR}")
endif()

if (NOT EXISTS "${TTMLIR_TOOLCHAIN_DIR}")
    message( FATAL_ERROR "The directory ${TTMLIR_TOOLCHAIN_DIR} does not exist. Please create it before running this script.\n  sudo mkdir -p ${TTMLIR_TOOLCHAIN_DIR}\n  sudo chown -R $ENV{USER} ${TTMLIR_TOOLCHAIN_DIR}")
endif()

# Install all python dependencies
add_custom_target(python-venv
    ALL COMMAND
    PYBUDA_PYTHON_VERSION=${PYBUDA_PYTHON_VERSION}
    PYBUDA_TOOLCHAIN_DIR=${PYBUDA_TOOLCHAIN_DIR}
    CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    PYBUDA_VENV_DIR=${PYBUDA_VENV_DIR}
    bash ${CMAKE_CURRENT_SOURCE_DIR}/create_venv.sh)
