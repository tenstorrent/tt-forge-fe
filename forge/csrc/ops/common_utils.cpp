// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common_utils.hpp"

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace common_utils
{

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> compute_elementwise_binary_shape(
    const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_ASSERT(in_shapes.size() == 2, "Elementwise binary ops should have exactly two input shapes.");

    std::vector<graphlib::DimBroadcast> broadcast;
    std::vector<std::uint32_t> output_shape;

    std::vector<std::uint32_t> shape0 = in_shapes[0];
    std::vector<std::uint32_t> shape1 = in_shapes[1];

    // Pad shorter shape with 1s at the beginning (rules of broadcast)
    while (shape0.size() < shape1.size())
    {
        shape0.insert(shape0.begin(), 1);
    }

    while (shape1.size() < shape0.size())
    {
        shape1.insert(shape1.begin(), 1);
    }

    // Adjust each dimension to match the broadcast rules
    for (size_t dim = 0; dim < shape0.size(); dim++)
    {
        if (shape0[dim] != shape1[dim])
        {
            if (shape0[dim] == 1)
            {
                // Broadcast first operand
                int negative_dim = static_cast<int>(dim) - static_cast<int>(shape0.size());
                broadcast.emplace_back(0, negative_dim, shape1[dim]);
                output_shape.push_back(shape1[dim]);
            }
            else
            {
                // shape1[dim] must be 1 for broadcast
                TT_ASSERT(
                    shape1[dim] == 1,
                    "Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to "
                    "broadcast");

                // Broadcast second operand
                int negative_dim = static_cast<int>(dim) - static_cast<int>(shape1.size());
                broadcast.emplace_back(1, negative_dim, shape0[dim]);
                output_shape.push_back(shape0[dim]);
            }
        }
        else
        {
            // Same dimension, no broadcast needed
            output_shape.push_back(shape0[dim]);
        }
    }

    return std::make_tuple(graphlib::Shape::create(output_shape), broadcast);
}

tt::graphlib::NodeContext reduce_broadcast_dimensions(
    tt::autograd::autograd_context &ac,
    const tt::graphlib::NodeContext &gradient,
    const tt::graphlib::Shape &input_shape,
    const tt::graphlib::Shape &grad_shape)
{
    // If shapes match, no reduction needed
    if (input_shape == grad_shape)
    {
        return gradient;
    }

    // Shapes don't match, we need to reduce along broadcast dimensions
    tt::graphlib::NodeContext result_grad = gradient;
    auto input_dims = input_shape.as_vector();
    auto grad_dims = grad_shape.as_vector();

    // Pad shapes with 1s at the beginning to match max rank
    size_t max_dims = std::max(input_dims.size(), grad_dims.size());

    std::vector<int> padded_input_dims(max_dims, 1);
    for (size_t i = 0; i < input_dims.size(); i++)
    {
        padded_input_dims[max_dims - input_dims.size() + i] = input_dims[i];
    }

    std::vector<int> padded_grad_dims(max_dims, 1);
    for (size_t i = 0; i < grad_dims.size(); i++)
    {
        padded_grad_dims[max_dims - grad_dims.size() + i] = grad_dims[i];
    }

    // For each dimension, if input_dim < grad_dim, we need to reduce_sum
    for (size_t i = 0; i < max_dims; i++)
    {
        if (padded_input_dims[i] >= padded_grad_dims[i])
            continue;

        int dim = static_cast<int>(i);
        Attrs named_attrs = {{"keep_dim", true}, {"dim_arg", dim}};
        result_grad =
            ac.autograd->create_op(ac, graphlib::OpType("reduce_sum", {dim, true}, named_attrs), {result_grad});
    }

    return result_grad;
}

}  // namespace common_utils
}  // namespace ops
}  // namespace tt
