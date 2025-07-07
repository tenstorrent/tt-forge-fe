// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>

#include "graph_lib/utils.hpp"
#include "ops/op.hpp"
#include "passes/passes_utils.hpp"

namespace tt::passes
{

using Attr = ForgeOpAttr;

static bool is_reshape(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "reshape";
}

template <typename T>
bool all_have_same_dim_and_shape_stride1(std::vector<T> const &v)
{
    if (v.size() == 0)
    {
        return false;
    }

    return std::all_of(
        v.begin(),
        v.end(),
        [&](T const &e)
        {
            auto attrs = dynamic_cast<graphlib::OpNode const *>(e)->op_legacy_attrs();
            int dim = std::get<int>(attrs[0]);
            return dim == std::get<int>(dynamic_cast<graphlib::OpNode const *>(v.front())->op_legacy_attrs()[0]) and
                   e->shape() == v.front()->shape() and std::get<int>(attrs[3]) == 1;
        });
}

/*
 * Decompose ND Reshape Split Pass
 *
 * Optimizes: Reshape (dimension split) → Index → Squeeze pattern
 * Converts to: Direct Index operations on original input
 *
 * EXAMPLE:
 *
 *   Input: {2, 12}
 *      |
 *   [Reshape] {2, 12} -> {2, 2, 6}  (splits dim 1: 12 -> 2×6)
 *      |
 *      +-- [Index dim=1, start=0, stop=1] -> {2, 1, 6} -> [Squeeze] -> {2, 6}
 *      +-- [Index dim=1, start=1, stop=2] -> {2, 1, 6} -> [Squeeze] -> {2, 6}
 *
 * BECOMES:
 *
 *   Input: {2, 12}
 *      |
 *      +-- [Index dim=1, start=0, stop=6] -> {2, 6}
 *      +-- [Index dim=1, start=6, stop=12] -> {2, 6}
 *
 * NOTE: The squeeze reshapes remain but become no-ops (input/output shapes match)
 *       and are removed by later passes like bypass_nop_tms.
 */

void decompose_nd_reshape_split(graphlib::Graph *graph)
{
    for (auto node : graph->nodes())
    {
        // Only consider reshape nodes
        if (not is_reshape(node))
            continue;

        // Only consider reshape which splits the dim into 2 dimensions
        // which means that difference between output and input rank must be 1
        auto reshape_producer = graph->operands(node)[0];
        uint32_t input_rank = reshape_producer->shape().size();
        uint32_t output_rank = node->shape().size();

        if (output_rank - input_rank != 1)
            continue;

        auto consumers = graph->users(node);
        bool all_consumers_are_index = all_of(
            consumers.begin(),
            consumers.end(),
            [](auto const &consumer)
            {
                auto op = dynamic_cast<graphlib::OpNode const *>(consumer);
                return op and op->op_name() == "index";
            });

        // All consumers must be index
        if (not all_consumers_are_index)
            continue;

        bool all_index_have_same_dim_and_shape = all_have_same_dim_and_shape_stride1(consumers);

        // All index must have same dim and shape
        if (not all_index_have_same_dim_and_shape)
            continue;

        uint32_t total_index_size = 0;
        int dim = std::get<int>(dynamic_cast<graphlib::OpNode const *>(consumers[0])->op_legacy_attrs()[0]);

        for (auto const &consumer : consumers)
        {
            total_index_size += consumer->shape()[dim];
        }
        bool index_all_channels = total_index_size == node->shape()[dim];
        bool all_index_have_length1 = all_of(
            consumers.begin(),
            consumers.end(),
            [](auto const &consumer)
            {
                auto op = dynamic_cast<graphlib::OpNode const *>(consumer);
                return std::get<int>(op->op_legacy_attrs()[2]) - std::get<int>(op->op_legacy_attrs()[1]) == 1;
            });

        // All index must have length 1 and total indexed size must be equal to node dim
        if (not(index_all_channels and all_index_have_length1))
            continue;

        bool all_sequence_consumers_are_squeeze = all_of(
            consumers.begin(),
            consumers.end(),
            [graph](auto const &consumer)
            {
                auto users = graph->users(consumer);
                auto shape_before = consumer->shape();
                auto shape_after = users[0]->shape();
                auto op = dynamic_cast<graphlib::OpNode const *>(users[0]);
                return users.size() == 1 and op->op_name() == "reshape" and
                       shape_after.volume() == shape_before.volume() and shape_after.size() + 1 == shape_before.size();
            });

        // All consumers of Index must be reshape nodes that are equivalent to squeeze
        if (not all_sequence_consumers_are_squeeze)
            continue;

        // find the original dim that was split
        std::optional<uint32_t> different_dim;
        for (uint32_t i = 0; i < input_rank; i++)
        {
            if (reshape_producer->shape()[i] != node->shape()[i])
            {
                different_dim = i;
                break;
            }
        }

        if (!different_dim.has_value())
            continue;

        uint32_t original_dim = different_dim.value();

        // validate that we found correct split dim
        uint32_t producer_dim_size = reshape_producer->shape()[original_dim];
        uint32_t expected_split_size = node->shape()[original_dim] * node->shape()[original_dim + 1];
        if (producer_dim_size != expected_split_size)
            continue;

        TT_ASSERT(producer_dim_size % total_index_size == 0);
        uint32_t new_dim_size = producer_dim_size / total_index_size;

        auto target_shape = reshape_producer->shape();
        target_shape[original_dim] = new_dim_size;

        // Remove reshape node and update index nodes to work like slice
        for (uint32_t i = 0; i < consumers.size(); i++)
        {
            auto op = dynamic_cast<graphlib::OpNode *>(consumers[i]);

            auto op_type_ = op->op_type();
            TT_ASSERT(op_type_.type() == ops::OpType::Index);
            int start = std::get<int>(op->op_type().legacy_attrs_[1]);

            // Update index attributes to slice the original tensor directly.
            // NOTE: since the old op infrastructure is used we need to set both the vector of attributes and the named
            // attributes. Once we transition to the new op infrastructure, only named attributes will be used.
            auto new_dim = static_cast<int>(original_dim);
            auto new_start = static_cast<int>(start * new_dim_size);
            auto new_stop = static_cast<int>(start * new_dim_size + new_dim_size);
            int new_stride = 1;

            std::vector<graphlib::OpType::Attr> new_attrs = {new_dim, new_start, new_stop, new_stride};

            op->change_op_type("index", new_attrs);
            op->set_op_attr("dim", new_dim);
            op->set_op_attr("start", new_start);
            op->set_op_attr("stop", new_stop);
            op->set_op_attr("stride", new_stride);

            op->set_shape(target_shape);
        }

        graphlib::bypass_node(graph, node, true);

        // Update node shapes in graph
        recalculate_shapes(graph);
    }
}

}  // namespace tt::passes
