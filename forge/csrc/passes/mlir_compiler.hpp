// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>

#include "tt/runtime/types.h"

namespace tt
{
    namespace graphlib
    {
        class Graph;
    }
}

namespace tt::passes
{
    /// Public API for running MLIR passes and generating binary.
    runtime::Binary run_mlir_compiler(tt::graphlib::Graph *graph);
}
