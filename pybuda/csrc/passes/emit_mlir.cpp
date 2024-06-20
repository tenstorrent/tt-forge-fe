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

namespace tt::passes
{


void emit_mlir(graphlib::Graph *graph)
{

    // Unary ops and Matmul SrcA can have reduced tile_dim

    for (auto *node : graphlib::topological_sort(*graph))
    {


        graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(node);
        std::cout << "Node name is " << node->name() << std::endl;

    }

}
}



