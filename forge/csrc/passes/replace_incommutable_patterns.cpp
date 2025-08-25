// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/replace_incommutable_patterns.hpp"

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "ops/op.hpp"
#include "passes/commute_utils.hpp"
#include "passes/passes_utils.hpp"
#include "passes/print_graph.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

// This function will hoist a bcast through a path of ops given that
// the final node in the path has one user edge which contains the bcast on dim.

static bool is_y_dim_concat_with_changed_x_dim(
    graphlib::Graph *graph, graphlib::OpNode *op, graphlib::Shape commute_shape)
{
    if (op->new_op_type() != ops::OpType::Concatenate)
        return false;

    int concat_dim = op->op_attr_as<int>("dim");

    if (concat_dim != -2)
        return false;

    if (commute_shape[-1] == op->shape()[-1])
        return false;

    // For now, lets make sure that all operands of the concat are a reshape
    // such that every dim of the operands of the reshapes are equivalent except for -2
    for (auto operand : graph->data_operands(op))
    {
        graphlib::OpNode *operand_op = dynamic_cast<graphlib::OpNode *>(operand);
        if (not operand_op or operand_op->new_op_type() != ops::OpType::Reshape)
            return false;

        auto operand_operand_shape = graph->data_operands(operand_op)[0]->shape();
        for (uint32_t i = 0; i < operand_operand_shape.size(); i++)
        {
            if (i == operand_operand_shape.size() - 2)
                continue;
            if (operand_operand_shape[i] != commute_shape[i])
                return false;
        }
    }

    return true;
}

static bool attempt_replace_downward_pattern(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape)
{
    if (is_y_dim_concat_with_changed_x_dim(graph, op, commute_shape))
    {
        // Place inverse reshapes on all operands
        auto operand_edges = graph->operand_data_edges(op);
        for (graphlib::Edge incoming_edge : operand_edges)
        {
            auto name = initial_op->name() + "_pattern_replacement_input_commute_clone" +
                        std::to_string(incoming_edge.edge_creation_id);
            auto *incoming_clone =
                graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
            graphlib::OpNode *incoming_clone_op = dynamic_cast<graphlib::OpNode *>(incoming_clone);
            auto incoming_clone_shape = commute_shape;
            int incoming_concat_dim_len = graph->node_by_id(incoming_edge.producer_node_id)->shape()[-2];
            incoming_clone_shape[-2] = incoming_concat_dim_len * clone_shape[-1] / commute_shape[-1];
            update_reshape_attr(incoming_clone_op, incoming_clone_shape);
            incoming_clone->set_shape(incoming_clone_shape);
            auto [edge_in, edge_out] = insert_node_on_edge(graph, incoming_edge, incoming_clone);
            // Set output df to match producer
            incoming_clone->set_output_df(graph->node_by_id(edge_in.producer_node_id)->output_df());
        }

        // Retrieve current op shape for output clones
        auto output_clone_shape = op->shape();

        // Convert op shape
        auto new_concat_shape = op->shape();
        new_concat_shape[-2] = op->shape()[-2] * op->shape()[-1] / commute_shape[-1];
        new_concat_shape[-1] = commute_shape[-1];
        op->set_shape(new_concat_shape);

        // Add golden transform
        std::vector<uint32_t> shape_vec = output_clone_shape.as_vector();
        std::vector<int> golden_transform_attrs;
        for (uint32_t d : shape_vec)
        {
            golden_transform_attrs.push_back((int)d);
        }
        op->add_golden_transform(graphlib::OpType("reshape", {{"shape", golden_transform_attrs}}));

        for (graphlib::Edge outgoing_edge : graph->user_data_edges(op))
        {
            auto name = initial_op->name() + "_pattern_replacement_output_commute_clone" +
                        std::to_string(outgoing_edge.edge_creation_id);
            auto *outgoing_clone =
                graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
            graphlib::OpNode *outgoing_clone_op = dynamic_cast<graphlib::OpNode *>(outgoing_clone);
            auto outgoing_clone_shape = output_clone_shape;
            update_reshape_attr(outgoing_clone_op, outgoing_clone_shape);
            outgoing_clone->set_shape(outgoing_clone_shape);
            auto [edge_in, edge_out] = insert_node_on_edge(graph, outgoing_edge, outgoing_clone, true, true, 0, true);
            // Set output df to match producer
            outgoing_clone->set_output_df(graph->node_by_id(edge_in.producer_node_id)->output_df());
        }
    }
    else
        return false;  // If we did not change a pattern then return false
    return true;
}

static bool attempt_replace_upward_pattern(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape)
{
    (void)graph;
    (void)initial_op;
    (void)op;
    (void)commute_shape;
    (void)clone_shape;
    return false;
}

static bool attempt_replace_pattern(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape,
    bool commute_up = false)
{
    if (not commute_up)
        return attempt_replace_downward_pattern(graph, initial_op, op, commute_shape, clone_shape);
    else
        return attempt_replace_upward_pattern(graph, initial_op, op, commute_shape, clone_shape);
}

static bool find_and_replace_incommutable_patterns(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    bool commute_up = false,
    graphlib::OpNode *from = nullptr,
    graphlib::OpNode *previous_op = nullptr)
{
    graphlib::OpNode *iter = from ? from : initial_op;
    auto clone_shape = initial_op->shape();

    bool replaced_pattern = false;
    while (not replaced_pattern)
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        log_trace(LogGraphCompiler, "  checking commute past {}", op->name());

        if (previous_op)
        {
            if (commute_up and are_bcasts_between_ops(graph, op, previous_op))
            {
                log_trace(
                    LogGraphCompiler,
                    "  Bcast between {} and {} prevents input commute",
                    op->name(),
                    previous_op->name());
                break;
            }
            else if (not commute_up)
                handle_shape_change_through_bcast(graph, initial_op, previous_op, op, &commute_shape, &clone_shape);
        }

        // If we've run into an inverse op along this path, then theres nothing to replace
        if (are_compatible_ops(graph, initial_op, op, &commute_shape))
        {
            break;
        }
        // TODO: (lpanos) I dont think is_elementwise should return true for any of these ops, but for now it does
        bool can_commute = op->is_eltwise() and op->new_op_type() != ops::OpType::Concatenate and
                           op->new_op_type() != ops::OpType::Select;

        if (not can_commute and op != initial_op)
        {
            if (attempt_replace_pattern(graph, initial_op, op, commute_shape, clone_shape, commute_up))
            {
                replaced_pattern = true;
            }
            break;
        }

        std::vector<graphlib::Node *> next_nodes = commute_up ? graph->data_operands(op) : graph->data_users(op);
        for (std::size_t i = 1; i < next_nodes.size(); ++i)
        {
            graphlib::OpNode *next_node = dynamic_cast<graphlib::OpNode *>(next_nodes[i]);
            replaced_pattern |= next_node and find_and_replace_incommutable_patterns(
                                                  graph, initial_op, commute_shape, commute_up, next_node, op);
        }

        if (replaced_pattern)
            break;

        TT_ASSERT(next_nodes.size() > 0);
        if (not commute_up)
        {
            graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(next_nodes[0]);
            if (output)
                break;
        }
        else
        {
            graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(next_nodes[0]);
            if (input)
                break;
        }

        previous_op = op;
        iter = dynamic_cast<graphlib::OpNode *>(next_nodes[0]);
        if (not iter)
            break;
    }

    return replaced_pattern;
}

bool replace_incommutable_patterns(graphlib::Graph *graph)
{
    bool updated_anything = false;
    // return false; // TODO Enable later
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        if (op->new_op_type() != ops::OpType::Reshape)
            continue;

        if (not find_and_replace_incommutable_patterns(graph, op, shape_of_only_operand(graph, op)))
        {
            if (not find_and_replace_incommutable_patterns(graph, op, shape_of_only_operand(graph, op), true))
                continue;
        }
        updated_anything = true;
        break;
    }
    reportify::dump_graph(graph->name(), "replace_incommutable_patterns", graph);
    return updated_anything;
}

}  // namespace tt::passes
