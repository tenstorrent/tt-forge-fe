// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_lowering_passes.hpp"

#include "graph_lib/utils.hpp"
#include "ops/op.hpp"
#include "passes/commute_utils.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"

namespace tt
{

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;

// Helper function to determine if a node is user-defined or compiler-defined
bool is_user_defined_node(const graphlib::Node *node)
{
    if (!node)
        return false;

    const graphlib::TaggedNode *tagged_node = node->as<graphlib::TaggedNode>();

    // Check for compiler decomposition tags
    if (tagged_node->has_tag("dont_decompose") || tagged_node->has_tag("optimize_hoist"))
    {
        return false;  // Compiler-defined due to decomposition tags
    }

    // Check naming pattern for compiler-generated nodes with .dc. suffix
    std::string node_name = node->name();
    if (node_name.find(".dc.") != std::string::npos)
    {
        return false;  // Compiler-defined due to naming pattern
    }

    // Check for frontend transformation using layer tag
    if (tagged_node->has_tag("layer"))
    {
        std::string layer_name = std::get<std::string>(tagged_node->tag_value("layer"));

        // if the prefix is / the node is Transformed from Frontend
        if (!layer_name.empty() && layer_name[0] == '/')
        {
            return false;  // Frontend transformation
        }
    }

    return true;
}
void convert_broadcast_ops_to_tms(Graph *graph)
{
    // Determines if the consumer node is unary and if the broadcast is user-defined
    // to decide whether to bypass the broadcast node.
    //
    // Bypass logic:
    // - is_consumer_unary(True) && user_defined(True) -> Don't bypass
    // - is_consumer_unary(False) && user_defined(True) -> Don't bypass
    // - is_consumer_unary(False) && user_defined(False) -> Bypass
    // - is_consumer_unary(True) && user_defined(False) -> Don't bypass

    std::vector<graphlib::Node *> broadcast_ops = graph->nodes(
        [](graphlib::Node *node) -> bool
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            return op and op->op_type() == ops::OpType::Broadcast;
        });

    for (Node *node : broadcast_ops)
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        auto op_type = op->op_type();
        bool has_unary_consumer = false;
        bool is_broadcast_user_defined = is_user_defined_node(node);

        // Check if any consumer is unary
        for (graphlib::Edge user_edge : graph->user_data_edges(node))
        {
            graphlib::OpNode *consumer_op =
                dynamic_cast<graphlib::OpNode *>(graph->node_by_id(user_edge.consumer_node_id));
            if (consumer_op && consumer_op->is_eltwise_unary())
            {
                has_unary_consumer = true;
                break;
            }
        }

        // Only bypass when: is_consumer_unary(False) && user_defined(False)
        bool should_bypass = !has_unary_consumer && !is_broadcast_user_defined;

        if (!should_bypass)
            continue;

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

    // Any unary op that doesn't change shape, and not transpose (which could keep the same shape)
    if (operand->as<graphlib::PyOpNode>()->op_type() == ops::OpType::Transpose)
        return false;

    graphlib::Shape incoming_shape = graph->data_operands(operand)[0]->shape();
    graphlib::Shape my_shape = operand->shape();

    return (my_shape.as_vector() == incoming_shape.as_vector());
}

}  // namespace tt
