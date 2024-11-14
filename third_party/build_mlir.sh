#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

source env/activate

build_type=${BUILD_TYPE:-Release}
c_compiler=${C_COMPILER:-clang}
cxx_compiler=${CXX_COMPILER:-clang++}
tt_runtime_debug=${TTMLIR_RUNTIME_DEBUG:-OFF}

source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTTMLIR_ENABLE_RUNTIME=ON -DTT_RUNTIME_DEBUG=$tt_runtime_debug

cmake --build build
