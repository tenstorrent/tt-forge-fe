// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "remove_nops.hpp"

#include <utils/logger.hpp>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"

namespace tt::passes
{

void remove_nops(graphlib::Graph *graph)
{
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        if (op->op_name() == "nop")
        {
            graphlib::bypass_node(graph, node, true);
        }
    }
}

}  // namespace tt::passes
