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
namespace divide
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 2, "OpDivide::eval should have two input tensors.");
    return torch::div(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Divide, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "divide::shape should have two input shapes.");
    TT_ASSERT(op.attrs().size() == 0, "divide::shape should not have any attrs.");

    return op_common::compute_elementwise_binary_shape(in_shapes);
}

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_ASSERT(inputs.size() == 2, "Divide should have exactly 2 inputs");
    TT_ASSERT(operand >= 0 && operand < 2, "Invalid operand index.");

    // if operand is 0, d/da(a/b) = 1/b
    tt::graphlib::NodeContext op_grad = ac.autograd->create_op(ac, graphlib::OpType("divide"), {gradient, inputs[1]});

    if (operand == 1)
    {
        // d/db(a/b) = -a/b² * grad
        // Step 1: Calculate b²
        auto b_squared = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {inputs[1], inputs[1]});
        // Step 2: Calculate a / b²
        auto a_over_b_squared = ac.autograd->create_op(ac, graphlib::OpType("divide"), {inputs[0], b_squared});
        // Step 3: Multiply by grad
        auto temp = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {gradient, a_over_b_squared});
        // Step 4: Negate the result
        auto neg_one = ac.autograd->create_constant(ac, -1.0f);
        op_grad = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {temp, neg_one});
    }

    // Reduce dimensions where broadcasting occurred using reduce_sum
    return op_common::reduce_broadcast_dimensions(ac, op_grad, inputs[operand].shape, gradient.shape);
}

}  // namespace divide

}  // namespace ops

}  // namespace tt
