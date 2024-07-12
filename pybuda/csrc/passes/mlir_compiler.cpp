// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_compiler.hpp"
#include "lower_to_mlir.hpp"
#include "mlir_passes.hpp"

// PyBuda headers
#include "graph_lib/graph.hpp"

// MLIR headers
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"

// TTMLIR headers
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToSerializedBinary.h"

namespace tt::passes
{
    /// Public API for lowering to MLIR, running MLIR passes and generate runtime binary.
    std::shared_ptr<void> run_mlir_compiler(tt::graphlib::Graph *graph)
    {
        // Register all the required dialects.
        mlir::DialectRegistry registry;
            
        registry.insert<
            mlir::tt::TTDialect, mlir::tt::ttir::TTIRDialect,
            mlir::tt::ttnn::TTNNDialect, mlir::arith::ArithDialect,
            mlir::func::FuncDialect, mlir::ml_program::MLProgramDialect,
            mlir::tensor::TensorDialect, mlir::emitc::EmitCDialect>();

        // Create a context with all registered dialects.
        mlir::MLIRContext context(registry);
        // Load all available dialects
        context.loadAllAvailableDialects();

        // Generate MLIR from the PyBuda graph.
        mlir::OwningOpRef<mlir::ModuleOp> mlir_module = lower_to_mlir(graph, context);

        // Run MLIR registered passes.
        run_mlir_passes(mlir_module);

        // Generate binary from the MLIR module.
        auto binary = mlir::tt::ttnn::emitTTNNAsFlatbuffer(mlir_module);

        if (binary == nullptr)
        {
            throw std::runtime_error("Failed to generate flatbuffer binary."); 
        }
        return binary;
    }
}