// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/materialize_unary_broadcasts.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

using Edge = graphlib::Edge;

// Collect edges from unary operations that have broadcast TMs
std::vector<Edge> collect_unary_broadcast_edges(Graph *graph)
{
    std::vector<Edge> edges_to_process;
    
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
    
    return edges_to_process;
}

// Create a broadcast operation for a specific edge and TM
Node* create_broadcast_op(Graph *graph, Node *producer, Node *consumer, 
                          const graphlib::OpType &tm, Node *current_node, int current_port)
{
    // Get broadcast dimension and size
    int dim = std::get<int>(tm.attr[0]);
    int size = std::get<int>(tm.attr[1]);
    
    // Create broadcast op node
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
    
    // Insert new node in the chain
    graph->add_edge(current_node, broadcast_op, current_port, 0);
    
    log_trace(LogGraphCompiler, "Created broadcast node: {} with dim={}, size={}", 
              new_node_name, dim, size);
    
    return broadcast_op;
}

void materialize_unary_broadcasts(Graph *graph)
{
    log_trace(LogGraphCompiler, "---------------------------------------------------");
    log_trace(LogGraphCompiler, "Starting materialize_unary_broadcasts pass...");
    
    // Find all edges with input_tms that can't be implicitly broadcasted
    // and convert them to explicit broadcast operations
    std::vector<Edge> edges_to_process = collect_unary_broadcast_edges(graph);
    
    // Process edges and create explicit broadcast operations
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
        
        log_trace(LogGraphCompiler, "Processing edge: {} -> {} with {} TMs", 
                  producer->name(), consumer->name(), tms.size());
        
        // Keep track of current node in the chain
        Node *current_node = producer;
        int current_port = edge.producer_output_port_id;
        
        // Create a chain of broadcast ops for each TM
        for (const auto &tm : tms)
        {
            if (tm.op != "broadcast")
                continue;
            
            current_node = create_broadcast_op(graph, producer, consumer, tm, current_node, current_port);
            current_port = 0;
        }
        
        // Connect the last broadcast op to the consumer
        graph->add_edge(current_node, consumer, current_port, edge.consumer_input_port_id);
        
        // Remove the original edge
        graph->remove_edge(edge);
    }
}

}  // namespace tt::passes