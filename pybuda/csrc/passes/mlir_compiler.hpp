// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>

namespace tt::graphlib
{
    class Graph;
}

namespace tt::passes
{
    /// Public API for running MLIR passes and generating binary.
    std::shared_ptr<void> run_mlir_compiler(tt::graphlib::Graph *graph);
}