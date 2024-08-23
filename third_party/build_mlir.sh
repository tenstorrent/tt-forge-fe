#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

source env/activate

build_type=${BUILD_TYPE:-Release}
c_compiler=${C_COMPILER:-clang}
cxx_compiler=${CXX_COMPILER:-clang++}
enable_runtime=${TTMLIR_ENABLE_RUNTIME:-OFF}
enable_perf_trace=${TT_RUNTIME_ENABLE_PERF_TRACE:-OFF}

source env/activate
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_C_COMPILER="$c_compiler" \
  -DCMAKE_CXX_COMPILER="$cxx_compiler" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTTMLIR_ENABLE_RUNTIME="$enable_runtime" \
  -DTT_RUNTIME_ENABLE_PERF_TRACE="$enable_perf_trace"

cmake --build build
