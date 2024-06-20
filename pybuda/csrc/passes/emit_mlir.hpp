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
namespace tt::passes 
{
void emit_mlir(graphlib::Graph *graph);
} // namespace tt:passes

