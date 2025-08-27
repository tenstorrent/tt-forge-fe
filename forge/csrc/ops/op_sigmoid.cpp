// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace sigmoid
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Sigmoid, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Sigmoid should have one input");

    return torch::sigmoid(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Sigmoid, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Sigmoid should have one input");
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
    TT_DBG_ASSERT(op.type() == OpType::Sigmoid, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Sigmoid should have one input");
    TT_ASSERT(operand == 0, "Invalid operand index");

    // dx = sigmoid(x) * (1 - sigmoid(x)) * grad

    auto one = ac.autograd->create_constant(ac, 1.0);

    auto sigm_ = ac.autograd->create_op(ac, Op("subtract"), {one, output});
    auto dsigm = ac.autograd->create_op(ac, Op("multiply"), {output, sigm_});

    return ac.autograd->create_op(ac, Op("multiply"), {dsigm, gradient});
}

}  // namespace sigmoid
}  // namespace ops
}  // namespace tt
