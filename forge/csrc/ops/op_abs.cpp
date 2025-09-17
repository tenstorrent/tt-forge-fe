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
namespace abs
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Abs, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "OpAbs::eval should have single input tensor.");
    return torch::abs(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Abs, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "OpAbs::shape should have single input shape.");
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
    TT_DBG_ASSERT(op.type() == OpType::Abs, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Abs should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    auto hs = ac.autograd->create_constant(ac, 0.5);
    auto heaviside = ac.autograd->create_op(ac, Op(OpType::Heaviside), {inputs[0], hs});

    auto st = ac.autograd->create_constant(ac, 0.5);
    auto subtract = ac.autograd->create_op(ac, Op(OpType::Subtract), {heaviside, st});

    auto sch = ac.autograd->create_constant(ac, 2.0);
    auto stretched = ac.autograd->create_op(ac, Op(OpType::Multiply), {subtract, sch});

    return ac.autograd->create_op(ac, Op(OpType::Multiply), {stretched, gradient});
}

}  // namespace abs
}  // namespace ops
}  // namespace tt
