// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "graph_lib/shape.hpp"
namespace tt::graphlib
{
class Graph;
class OpNode;
class Node;
class Shape;
}

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace tt::passes 
{
// Public API for generating MLIR from the PyBuda graph.
void emit_mlir(tt::graphlib::Graph * graph);
} // namespace tt:passes

