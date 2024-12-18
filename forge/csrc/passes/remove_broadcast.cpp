// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "remove_broadcast.hpp"

#include <utils/logger.hpp>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"

namespace tt::passes
{

void remove_broadcast(graphlib::Graph *graph)
{
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            std::vector<graphlib::Edge> data_edges = graph->operand_data_edges(node);
            for (const graphlib::Edge &data_edge : data_edges)
            {
                graph->get_edge_attributes(data_edge)->clear_broadcast_dims();
                graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);

                auto consumer = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(data_edge.consumer_node_id));
                auto user_edges = graph->user_data_edges(node);

                if (not op)
                    continue;

                if (op->op_name() == "broadcast")
                {
                    if (graphlib::is_eltwise_binary(consumer) or consumer->op_name() == "broadcast")
                    {
                        graphlib::bypass_node(graph, node, true);
                        updated = true;
                        break;
                    }
                }

                if (updated)
                    break;
            }
        }
    }
}

}  // namespace tt::passes
