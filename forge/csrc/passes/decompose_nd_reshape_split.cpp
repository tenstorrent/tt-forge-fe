// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>

#include "graph_lib/utils.hpp"
#include "ops/op.hpp"
#include "passes/passes_utils.hpp"

namespace tt::passes
{

static bool is_reshape(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_type() == ops::OpType::Reshape;
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
            auto op_node = dynamic_cast<graphlib::OpNode const *>(e);
            auto front_op_node = dynamic_cast<graphlib::OpNode const *>(v.front());
            int dim = op_node->op_attr_as<int>("dim");
            int stride = op_node->op_attr_as<int>("stride");
            int front_dim = front_op_node->op_attr_as<int>("dim");
            return dim == front_dim and e->shape() == v.front()->shape() and stride == 1;
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
                return op and op->op_type() == ops::OpType::Index;
            });

        // All consumers must be index
        if (not all_consumers_are_index)
            continue;

        bool all_index_have_same_dim_and_shape = all_have_same_dim_and_shape_stride1(consumers);

        // All index must have same dim and shape
        if (not all_index_have_same_dim_and_shape)
            continue;

        uint32_t total_index_size = 0;
        int dim = dynamic_cast<graphlib::OpNode const *>(consumers[0])->op_attr_as<int>("dim");

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
                int stop = op->op_attr_as<int>("stop");
                int start = op->op_attr_as<int>("start");
                return stop - start == 1;
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
                return users.size() == 1 and op->op_type() == ops::OpType::Reshape and
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

            auto op_type_ = op->op();
            TT_ASSERT(op_type_.type() == ops::OpType::Index);
            int start = op->op_attr_as<int>("start");

            // Update index attributes to slice the original tensor directly.
            // NOTE: since the old op infrastructure is used we need to set both the vector of attributes and the named
            // attributes. Once we transition to the new op infrastructure, only named attributes will be used.
            auto new_dim = static_cast<int>(original_dim);
            auto new_start = static_cast<int>(start * new_dim_size);
            auto new_stop = static_cast<int>(start * new_dim_size + new_dim_size);
            int new_stride = 1;

            op->change_op(
                ops::Op(ops::OpType::Index).as_string(),
                {{"dim", new_dim}, {"start", new_start}, {"stop", new_stop}, {"stride", new_stride}});

            op->set_shape(target_shape);
        }

        graphlib::bypass_node(graph, node, true);

        // Update node shapes in graph
        recalculate_shapes(graph);
    }
}

}  // namespace tt::passes
