// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/materialize_unary_broadcasts.hpp"

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/commute_utils.hpp"
#include "passes/passes_utils.hpp"

namespace tt::passes
{

using Edge = graphlib::Edge;

void materialize_unary_broadcasts(Graph *graph)
{
    // Find all edges with input_tms that can't be implicitly broadcasted
    // and convert them to explicit operations
    std::vector<Edge> edges_to_process;
    
    // First collect all edges with broadcast TMs
    for (Node *node : graph->nodes())
    {
        // Only consider unary operations (nodes with single input)
        if (graph->operand_data_edges(node).size() != 1)
            continue;
        
        for (Edge edge : graph->operand_data_edges(node))
        {
            auto attrs = graph->get_edge_attributes(edge);
            if (attrs->has_tms())
            {
                edges_to_process.push_back(edge);
            }
        }
    }
    
    // Process the edges and create explicit broadcast operations
    for (Edge edge : edges_to_process)
    {
        auto attrs = graph->get_edge_attributes(edge);
        if (!attrs->has_tms())
            continue;
        
        std::vector<graphlib::OpType> tms = attrs->get_tms();
        if (tms.empty())
            continue;
        
        Node *producer = graph->node_by_id(edge.producer_node_id);
        Node *consumer = graph->node_by_id(edge.consumer_node_id);

        graphlib::Shape producer_shape = producer->shape();
        
        // Keep track of the current node in the chain
        Node *current_node = producer;
        int current_port = edge.producer_output_port_id;
        
        // Create a chain of broadcast ops for each TM
        for (const auto &tm : tms)
        {
            if (tm.op != "broadcast")
                continue;
                
            // Get the broadcast dimension and size
            int dim = std::get<int>(tm.attr[0]);
            int size = std::get<int>(tm.attr[1]);
            
            // Skip trivial broadcasts (size 1)
            if (size == 1)
                continue;
            
            // Create a new broadcast op node
            std::string new_node_name = producer->name() + "_bcast_" + std::to_string(dim) + 
                                                            "_to_" + consumer->name();
            std::unique_ptr<graphlib::PyOpNode> broadcast_node = graphlib::create_node<graphlib::PyOpNode>(
                new_node_name, graphlib::OpType("broadcast", {dim, size, true}));
            graphlib::PyOpNode *broadcast_op = graph->add_node(
                std::move(broadcast_node),
                graph->get_subgraph_id_for_node(producer->id()));
            
            // Calculate the new shape after this broadcast
            graphlib::Shape new_shape = current_node->shape();
            new_shape[dim] = size;
            broadcast_op->set_shape(new_shape);
            broadcast_op->set_output_df(current_node->output_df());
            
            // Insert the new node in the chain
            graph->add_edge(current_node, broadcast_op, current_port, 0);
            
            // Update current node for next iteration
            current_node = broadcast_op;
            current_port = 0;
        }
        
        // Connect the last broadcast op to the consumer
        graph->add_edge(current_node, consumer, current_port, edge.consumer_input_port_id);
        
        // Remove the original edge
        graph->remove_edge(edge);
    }
}

}  // namespace tt::passes