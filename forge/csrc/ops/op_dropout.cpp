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
    TT_ASSERT(tensors.size() == 1, "Dropout should have one input.");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed.");

    float p = op.attr_as<float>("p");
    bool training = op.attr_as<bool>("training");
    int seed = op.attr_as<int>("seed");

    if (!training || p < 0.0)
        return tensors[0];

    at::Tensor result;

    // Simple C++ random number generation for dropout mask
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    at::Tensor mask = at::ones_like(tensors[0]);
    auto mask_accessor = mask.accessor<float, 2>();

    for (int i = 0; i < mask.size(0); i++)
        for (int j = 0; j < mask.size(1); j++) mask_accessor[i][j] = dis(gen) >= p ? (1.0f / (1.0f - p)) : 0.0f;

    return tensors[0] * mask;
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Dropout, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Dropout should have one input.");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed.");

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
    TT_ASSERT(inputs.size() == 1, "Dropout should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for dropout.");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed.");

    // Apply dropout to gradient with the same parameters.
    return ac.autograd->create_op(
        ac,
        graphlib::OpType(
            "dropout",
            {},
            {{"p", op.attr_as<float>("p")},
             {"training", op.attr_as<bool>("training")},
             {"seed", op.attr_as<int>("seed")}}),
        {gradient});
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_ASSERT(inputs.size() == 1, "Dropout should have one input");

    // Dropout has minimal computational cost - just element-wise masking
    // FLOPS = number of elements (for mask generation and application).
    long flops = 1;
    for (auto dim : inputs[0]) flops *= dim;

    return flops;
}

}  // namespace dropout
}  // namespace ops
}  // namespace tt
