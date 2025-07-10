// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "lower_to_forge/common.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace heaviside
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Heaviside, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "Eltwise binary should have two inputs");

    return torch::heaviside(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Heaviside, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "Eltwise binary should have two inputs");

    std::vector<graphlib::DimBroadcastTrampoline> broadcast;
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
            if (shape1[dim] == 1)
            {
                // Broadcast second operand
                int negative_dim = static_cast<int>(dim) - static_cast<int>(shape1.size());
                broadcast.emplace_back(1, negative_dim, shape0[dim]);
                output_shape.push_back(shape0[dim]);
            }
            else
            {
                // shape0[dim] must be 1 for broadcast
                TT_ASSERT(
                    shape0[dim] == 1,
                    "Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to "
                    "broadcast");

                // Broadcast first operand
                int negative_dim = static_cast<int>(dim) - static_cast<int>(shape0.size());
                broadcast.emplace_back(0, negative_dim, shape1[dim]);
                output_shape.push_back(shape1[dim]);
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

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Heaviside, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "Eltwise binary should have two inputs");

    // Decompose Heaviside: result = (x > 0) + (x == 0) * y
    auto x = inputs[0];
    auto y = inputs[1];

    // Create zero constant
    auto zero = dc.op(graphlib::OpType("constant", {}, {{"c", 0.0f}}), {});

    // x > 0
    auto x_gt = dc.op(graphlib::OpType("greater"), {x, zero});

    // x == 0
    auto x_eq = dc.op(graphlib::OpType("equal"), {x, zero});

    // (x == 0) * y
    auto res = dc.op(graphlib::OpType("multiply"), {x_eq, y});

    // (x > 0) + (x == 0) * y
    res = dc.op(graphlib::OpType("add"), {res, x_gt});

    dc.fuse(res, 0);
}

}  // namespace heaviside
}  // namespace ops
}  // namespace tt
