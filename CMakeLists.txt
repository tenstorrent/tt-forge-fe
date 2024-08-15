cmake_minimum_required(VERSION 3.20.0)
project(ttforge LANGUAGES CXX)

find_program(CLANG_17 clang++-17)
find_program(CLANG clang)
if(CLANG_17 AND CLANG)
    message(STATUS "Found Clang-17 here: ${CLANG_17}")
    set(CMAKE_CXX_COMPILER "${CLANG_17}")
    set(CMAKE_C_COMPILER "${CLANG}")
else()
    message(FATAL_ERROR "Clang++-17 or clang not found!!!")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(Utils)

check_required_env_var(TTFORGE_PYTHON_VERSION)
check_required_env_var(TTFORGE_TOOLCHAIN_DIR)
check_required_env_var(TTFORGE_VENV_DIR)

set(TTFORGE_PYTHON_VERSION $ENV{TTFORGE_PYTHON_VERSION})
set(TTFORGE_TOOLCHAIN_DIR $ENV{TTFORGE_TOOLCHAIN_DIR})
set(TTFORGE_VENV_DIR $ENV{TTFORGE_VENV_DIR})

set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
set(TTMLIR_VENV_DIR $ENV{TTMLIR_VENV_DIR})

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-DDEBUG)
endif()

set(TTFORGE_CSRC_WARNINGS -Wall -Wextra -Wno-pragmas -Wno-unused-parameter)
set(CFLAGS_NO_WARN -DFMT_HEADER_ONLY)
set(TTFORGE_CSRC_CFLAGS ${CFLAGS_NO_WARN} ${TTFORGE_CSRC_WARNINGS} -DUTILS_LOGGER_PYTHON_OSTREAM_REDIRECT=1)

set(CONFIG_LDFLAGS "")

set(STATIC_LIB_FLAGS -fPIC)
set(SHARED_LIB_FLAGS -fPIC)

add_subdirectory(third_party)
add_subdirectory(pybuda)
add_subdirectory(docs)

### Generate stubs for ttforge
### Run `cmake --build build -- make_stubs` to generate stubs
add_custom_target(make_stubs
    COMMAND pip install mypy==1.10
    COMMAND stubgen -m pybuda._C -m pybuda._C.autograd -m pybuda._C.graph -m pybuda._C.torch_device -m pybuda._C.runtime -o pybuda -v
    COMMENT "Generating stubs for ttforge"
    USES_TERMINAL
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
