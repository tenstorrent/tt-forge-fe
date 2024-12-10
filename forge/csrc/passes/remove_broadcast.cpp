
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes {
void remove_broadcast(graphlib::Graph *graph)
{
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            // Skip nodes that are not operation nodes
            auto *op_node = dynamic_cast<graphlib::OpNode *>(node);
            if (!op_node)
            {
                continue;
            }
            auto user_edges = graph->user_data_edges(node);
            if (user_edges.size() != 1)
            {
                continue;
            }

            graphlib::Edge edge = user_edges[0];
            auto edge_attrs = graph->get_edge_attributes(edge);

            // Identify broadcast either by edge attributes or node type
            if (op_node->op_name() != "broadcast")
            {   
                continue;
            }
            // Ensure the node has exactly one operand edge before bypassing
            auto op_edges = graph->operand_edges(node);
            if (op_edges.size() != 1)
            {
                log_debug(LogGraphCompiler, "Skipping node {} as it has {} operand edges.", node->name(), op_edges.size());
                continue;
            }
            log_debug(LogGraphCompiler, "Removing broadcast node: {}", node->name());
            bypass_node(graph, node, true);
            updated = true;
            break;
        }
    }
}
}