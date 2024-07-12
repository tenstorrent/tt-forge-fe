// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_passes.hpp"

// Standard headers
#include <stdexcept>

// MLIR headers
#include "mlir/IR/BuiltinOps.h"

// TTMLIR headers
#include "ttmlir/Dialect/TTIR/Passes.h"
#include "ttmlir/Dialect/TTNN/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

namespace tt::passes
{
    /// Public API for running MLIR passes and generating binary.
    void run_mlir_passes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module)
    {
        // Register required passes
        mlir::tt::ttir::registerPasses();
        mlir::tt::ttnn::registerPasses();

        // Create a pass manager.
        mlir::PassManager pm(mlir_module.get()->getName());

        // Create a pass pipeline
        mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(pm);

        // Run the pass manager.
        if (mlir::failed(pm.run(mlir_module.get())))
        {
            throw std::runtime_error("Failed to run MLIR compiler pass pipeline.");
        }

        mlir_module.get().dump();
    }
}