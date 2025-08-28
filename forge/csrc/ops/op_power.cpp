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
namespace power
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Power, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "OpPower::eval should have two input tensors.");
    return torch::pow(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Power, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "OpPower::shape should have two input shapes.");

    // Power operation should broadcast the two input shapes
    return op_common::compute_elementwise_binary_shape(in_shapes);
}

tt::graphlib::NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Power, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "Power should have two inputs.");
    TT_ASSERT(operand == 0 || operand == 1, "Invalid operand index.");

    tt::graphlib::NodeContext op_grad = gradient;

    if (operand == 0)
    {
        // dx = y * (x^y) * recp(x) = y * output / x (this approach might be numerically unstable because of the
        // division by x which can be zero)
        auto recip = ac.autograd->create_op(ac, Op("reciprocal"), {inputs[0]});
        auto partial_grad = ac.autograd->create_op(ac, Op("multiply"), {output, recip});
        auto pow_grad = ac.autograd->create_op(ac, Op("multiply"), {inputs[1], partial_grad});
        op_grad = ac.autograd->create_op(ac, Op("multiply"), {pow_grad, gradient});
    }
    else
    {
        // dy = (x^y) * ln(x) = output * ln(x)
        auto ln_x = ac.autograd->create_op(ac, Op("log"), {inputs[0]});
        auto pow_grad = ac.autograd->create_op(ac, Op("multiply"), {output, ln_x});
        op_grad = ac.autograd->create_op(ac, Op("multiply"), {pow_grad, gradient});
    }

    // Reduce dimensions where broadcasting occurred using reduce_sum
    return op_common::reduce_broadcast_dimensions(ac, op_grad, inputs[operand].shape, gradient.shape);
}

}  // namespace power
}  // namespace ops
}  // namespace tt
