// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "forge_graph_module.hpp"

namespace tt::graphlib
{
    class Graph;
}

namespace tt::passes 
{
    // Split the graph into multiple graphs which will lower to different MLIR programs,
    // i.e. forward, backward, etc.
    ForgeGraphModule split_graph(tt::graphlib::Graph* graph);

} // namespace tt:passes

