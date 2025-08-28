// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "ops/op_common.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace atan
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Atan, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Atan should have one input");
    return torch::atan(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Atan, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Atan should have one input");
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    /**
     * Derivative of atan(x) = 1 / (1 + x^2)
     */

    TT_DBG_ASSERT(op.type() == OpType::Atan, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Atan should have one input");
    TT_ASSERT(operand == 0, "Invalid operand index");

    // x^2
    auto x_squared = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], inputs[0]});

    // 1 + x^2
    auto one = ac.autograd->create_constant(ac, 1.0);
    auto one_plus_x_squared = ac.autograd->create_op(ac, Op(OpType::Add), {one, x_squared});

    // 1 / (1 + x^2)
    auto derivative = ac.autograd->create_op(ac, Op(OpType::Divide), {one, one_plus_x_squared});

    // derivative * gradient
    return ac.autograd->create_op(ac, Op(OpType::Multiply), {derivative, gradient});
}

}  // namespace atan
}  // namespace ops
}  // namespace tt
