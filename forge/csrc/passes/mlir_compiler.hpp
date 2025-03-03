// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tt/runtime/types.h"

namespace tt
{
class ForgeGraphModule;
}

namespace tt::passes
{
/// Public API for running MLIR passes and generating binary.
runtime::Binary run_mlir_compiler(tt::ForgeGraphModule& module);

/// Public API for lowering to MLIR, running MLIR passes and generate C++ code.
std::string run_mlir_compiler_to_cpp(tt::ForgeGraphModule& module);
}  // namespace tt::passes
