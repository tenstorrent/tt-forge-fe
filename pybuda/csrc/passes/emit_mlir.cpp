// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/emit_mlir.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/Passes.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Passes.h"

namespace tt::passes
{


mlir::FloatType float_type(graphlib::Node *node, mlir::OpBuilder &builder)
{
    switch (node->output_df())
    {
    case tt::DataFormat::Float32:
        return builder.getF32Type();
    case tt::DataFormat::Float16_b:
        return builder.getF16Type();
    }
    assert(false);
}

mlir::Type get_node_type(graphlib::Node *node, mlir::OpBuilder &builder)
{
    std::vector<int64_t> shape_vec;
    for (auto dim : node->shape().as_vector())
    {
        shape_vec.push_back((int64_t)dim);
    }
    return mlir::RankedTensorType::get(shape_vec, float_type(node, builder));
}



void emit_mlir(graphlib::Graph *graph)
{
    mlir::DialectRegistry registry;
    registry.insert<
            mlir::tt::TTDialect, mlir::tt::ttir::TTIRDialect,
            mlir::arith::ArithDialect, mlir::func::FuncDialect,
            mlir::ml_program::MLProgramDialect, mlir::tensor::TensorDialect>();

    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();
    mlir::OpBuilder builder(&ctx);
    auto module = builder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(&ctx), "pybuda_graph");
    auto &moduleBlock = module.getBodyRegion().front();

    builder.setInsertionPointToStart(&moduleBlock);

    // assemble the inputs
    llvm::SmallVector<mlir::Type> inputs;

    for (auto *input : graph->nodes_by_type(tt::graphlib::kInput))
        inputs.push_back(get_node_type(input, builder));

    // assemble the outputs
    llvm::SmallVector<mlir::Type> outputs;
    for (auto *output : graph->nodes_by_type(tt::graphlib::kOutput))
        outputs.push_back(get_node_type(output, builder));
    
    auto funcType = builder.getType<mlir::FunctionType>(mlir::TypeRange(inputs), mlir::TypeRange(outputs));
    auto func = builder.create<mlir::func::FuncOp>(module.getLoc(), "main", funcType);
    mlir::Block *entryBlock = func.addEntryBlock();
    // entryBlock->addArguments(inputs, llvm::SmallVector<mlir::Location, 4>(inputs.size(), mlir::UnknownLoc()));

    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(node);
        // convert_op(op_node, builder, entryBlock);
    }

    module.dump();
}
}



