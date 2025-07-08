// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace divide
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 2, "OpDivide::eval should have two input tensors.");
    return torch::div(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_ASSERT(in_shapes.size() == 2, "OpAdd::shape should have two input shapes.");

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

}  // namespace divide

}  // namespace ops

}  // namespace tt
