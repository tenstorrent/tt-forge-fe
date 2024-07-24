// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_compiler.hpp"
#include <memory>
#include "lower_to_mlir.hpp"
#include "mlir_passes.hpp"

// PyBuda headers
#include "graph_lib/graph.hpp"

// MLIR headers
#include "mlir/IR/BuiltinOps.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#include "mlir/InitAllDialects.h"
#pragma clang diagnostic pop

// TTMLIR headers
#include "tt/runtime/types.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include "tt_torch_device/tt_device.hpp"

namespace tt::passes
{
    /// Public API for lowering to MLIR, running MLIR passes and generate runtime binary.
    runtime::Binary run_mlir_compiler(tt::graphlib::Graph *graph)
    {
        // Register all the required dialects.
        mlir::DialectRegistry registry;
            
        registry.insert<
            mlir::tt::TTDialect, mlir::tt::ttir::TTIRDialect,
            mlir::tt::ttnn::TTNNDialect, mlir::arith::ArithDialect,
            mlir::func::FuncDialect, mlir::ml_program::MLProgramDialect,
            mlir::tensor::TensorDialect>();

        // Create a context with all registered dialects.
        mlir::MLIRContext context(registry);
        // Load all available dialects
        context.loadAllAvailableDialects();

        // Generate MLIR from the PyBuda graph.
        mlir::OwningOpRef<mlir::ModuleOp> mlir_module = lower_to_mlir(graph, context);
        tt::log_info("MLIR module generated successfully.");

        // Run MLIR registered passes.
        run_mlir_passes(mlir_module);
        tt::log_info("MLIR passes run successfully.");

        // Generate binary from the MLIR module.
        auto binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());
        tt::log_info("Flatbuffer binary generated successfully.");

        if (binary == nullptr)
        {
            throw std::runtime_error("Failed to generate flatbuffer binary."); 
        }

        return binary;
    }
}
