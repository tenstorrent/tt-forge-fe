// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_lowering_passes.hpp"

#include "graph_lib/utils.hpp"
#include "passes/commute_utils.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"

namespace tt
{

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;

void convert_broadcast_ops_to_tms(Graph *graph)
{
    std::vector<Node *> broadcast_ops = graph->nodes(
        [](Node *node) -> bool
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            return op and op->op_name() == "broadcast";
        });

    for (Node *node : broadcast_ops)
    {
        graphlib::OpNode *op = node->as<graphlib::OpNode>();
        graphlib::OpType op_type = op->op_type();
        constexpr bool remove_node = true;
        graphlib::bypass_node(
            graph,
            node,
            remove_node,
            [graph, op_type](Edge new_edge, Edge)
            {
                auto attr = graph->get_edge_attributes(new_edge);
                attr->prepend_tm(op_type);
            });
    }
}

void place_inter_subgraph_queues(graphlib::Graph *graph)
{
    for (Node *n : graph->nodes_by_type(NodeType::kOutput))
    {
        std::vector<Node *> consumers = graph->data_users(n);
        if (consumers.size() == 0)
            continue;
        std::vector<Node *> producers = graph->data_operands(n);
        TT_ASSERT(producers.size() == 1);

        std::cout << "removing node: " << n->name() << std::endl;
        graph->remove_node(n);
        for (Node *consumer : consumers)
        {
            std::cout << "adding edge from: " << producers[0]->name() << " to: " << consumer->name() << std::endl;
            graph->add_edge(producers[0], consumer);
        }
    }
}

void replace_with_broadcasted_const(
    Graph *graph,
    graphlib::ConstantInputNode *constant,
    std::shared_ptr<void> broadcasted_tensor,
    graphlib::Shape target_shape,
    graphlib::PyOpNode *original_tile_bcast)
{
    auto broadcasted_const = graph->add_node(
        graphlib::create_node<graphlib::ConstantInputNode>(
            constant->name() + "_tile_bcast", broadcasted_tensor, target_shape),
        graph->get_subgraph_id_for_node(constant->id()));
    broadcasted_const->set_shape(target_shape);
    broadcasted_const->set_output_df(original_tile_bcast->output_df());
    graphlib::Edge edge(broadcasted_const->id(), 0, original_tile_bcast->id(), 0, graphlib::EdgeType::kData);
    graph->add_edge(edge);
    graph->remove_node(constant);
}

bool safe_to_hoist_past(const Graph *graph, const Node *operand)
{
    if (graph->user_data_edges(operand).size() > 1)
        return false;  // we don't want to deal with this now

    if (graph->operand_data_edges(operand).size() > 1)
        return false;  // not a unary op

    if (operand->node_type() != NodeType::kPyOp)
        return false;

    const std::string &op_type = operand->as<graphlib::PyOpNode>()->op_type().name();

    // Any unary op that doesn't change shape, and not transpose (which could keep the same shape)
    if (op_type == "transpose")
        return false;

    graphlib::Shape incoming_shape = graph->data_operands(operand)[0]->shape();
    graphlib::Shape my_shape = operand->shape();

    return (my_shape.as_vector() == incoming_shape.as_vector());
}

}  // namespace tt
