// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <random>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "lower_to_forge/common.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace dropout
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Dropout, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Dropout should have one input");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed");

    // Extract attributes
    float p = op.attr_as<float>("p");
    bool training = op.attr_as<bool>("training");
    int seed = op.attr_as<int>("seed");

    // Apply dropout
    at::Tensor result;
    if (training && p > 0.0)
    {
        // Simple C++ random number generation for dropout mask
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        // Create dropout mask manually
        at::Tensor mask = at::ones_like(tensors[0]);
        auto mask_accessor = mask.accessor<float, 2>();

        for (int i = 0; i < mask.size(0); i++)
        {
            for (int j = 0; j < mask.size(1); j++)
            {
                if (dis(gen) < p)
                {
                    mask_accessor[i][j] = 0.0f;
                }
                else
                {
                    mask_accessor[i][j] = 1.0f / (1.0f - p);  // Scale by 1/(1-p)
                }
            }
        }

        result = tensors[0] * mask;
    }
    else
    {
        // In eval mode or p=0, just return input unchanged
        result = tensors[0];
    }

    return result;
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Dropout, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Dropout should have one input");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed");

    // Dropout preserves input shape
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcastTrampoline>{});
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Dropout, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Dropout should have one input");
    TT_ASSERT(operand == 0, "Invalid operand index for dropout");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed");

    // Extract attributes for backward pass
    float p = op.attr_as<float>("p");
    bool training = op.attr_as<bool>("training");
    int seed = op.attr_as<int>("seed");

    // Create named attributes for the backward dropout operation
    graphlib::OpType::Attrs attrs;
    attrs["p"] = p;
    attrs["training"] = training;
    attrs["seed"] = seed;

    // Apply dropout to gradient with same parameters
    return ac.autograd->create_op(ac, graphlib::OpType("dropout", {}, attrs), {gradient});
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_ASSERT(inputs.size() == 1, "Dropout should have one input");

    // Dropout has minimal computational cost - just element-wise masking
    // FLOPS = number of elements (for mask generation and application)
    long flops = 1;
    for (auto dim : inputs[0])
    {
        flops *= dim;
    }
    return flops;
}

}  // namespace dropout
}  // namespace ops
}  // namespace tt
