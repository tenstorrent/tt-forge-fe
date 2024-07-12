// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace tt::graphlib
{
class Graph;
}

namespace mlir {
    class MLIRContext;
    class ModuleOp;
    template <typename OpTy> class OwningOpRef;
} // namespace mlir

namespace tt::passes 
{
    // Public API for generating MLIR from the PyBuda graph.
    mlir::OwningOpRef<mlir::ModuleOp> lower_to_mlir(tt::graphlib::Graph * graph, mlir::MLIRContext& context);
} // namespace tt:passes

