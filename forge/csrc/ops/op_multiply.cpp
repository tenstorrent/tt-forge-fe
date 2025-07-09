// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "autograd/binding.hpp"
#include "common_utils.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace multiply
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Multiply, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "multiply::eval should have two input tensors.");
    TT_ASSERT(op.attrs().size() == 0, "multiply::eval should not have any attrs.");

    return torch::multiply(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    return common_utils::compute_elementwise_binary_shape(in_shapes);
}

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Multiply, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "multiply::decompose_post_autograd should have two input tensors.");
    TT_ASSERT(op.attrs().size() == 0, "multiply::decompose_post_autograd should not have any attrs.");

    // Get dimensions of both inputs
    Shape input0_shape = inputs[0].shape;
    Shape input1_shape = inputs[1].shape;

    uint32_t input0_size = input0_shape.size();
    uint32_t input1_size = input1_shape.size();

    if (input0_size > input1_size && input0_size == 5)
    {
        tt::graphlib::NodeContext ops1 =
            dc.op(graphlib::OpType("reshape", {}, {{"shape", input0_shape.as_vector<int>()}}), {inputs[1]});

        tt::graphlib::NodeContext result = dc.op(graphlib::OpType("multiply"), {inputs[0], ops1});

        dc.fuse(result, 0);
    }
    else if (input1_size > input0_size && input1_size == 5)
    {
        tt::graphlib::NodeContext ops0 =
            dc.op(graphlib::OpType("reshape", {}, {{"shape", input1_shape.as_vector<int>()}}), {inputs[0]});

        tt::graphlib::NodeContext result = dc.op(graphlib::OpType("multiply"), {ops0, inputs[1]});

        dc.fuse(result, 0);
    }
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Multiply, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "multiply::backward should have two input tensors.");
    TT_ASSERT(op.attrs().size() == 0, "multiply::backward should not have any attrs.");
    TT_ASSERT(operand >= 0 && operand < 2, "Invalid operand index.");

    // For multiply x * y: dx = grad * y, dy = grad * x
    NodeContext op_grad = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {gradient, inputs[1 - operand]});

    // If the shapes are the same, no need to reduce dimensions
    const graphlib::Shape &input_shape = inputs[operand].shape;
    const graphlib::Shape &grad_shape = gradient.shape;
    if (input_shape == grad_shape)
        return op_grad;

    // Reduce dimensions where broadcasting occurred using reduce_sum
    return common_utils::reduce_broadcast_dimensions(ac, op_grad, input_shape, grad_shape);
}

}  // namespace multiply
}  // namespace ops
}  // namespace tt
