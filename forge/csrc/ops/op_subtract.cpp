// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "autograd/binding.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace subtract
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Subtract, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "subtract::eval should have two input tensors.");
    TT_ASSERT(op.attrs().size() == 0, "subtract::eval should not have any attrs.");

    return torch::subtract(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Subtract, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "subtract::shape should have two input shapes.");
    TT_ASSERT(op.attrs().size() == 0, "subtract::shape should not have any attrs.");

    return op_common::compute_elementwise_binary_shape(in_shapes);
}

void decompose_post_autograd(
    const graphlib::OpType &old_op_type,
    const Op &op,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Subtract, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "subtract::decompose_post_autograd should have two input tensors.");
    TT_ASSERT(op.attrs().size() == 0, "subtract::decompose_post_autograd should not have any attrs.");

    // Get dimensions of both inputs
    Shape input0_shape = inputs[0].shape;
    Shape input1_shape = inputs[1].shape;

    uint32_t input0_size = input0_shape.size();
    uint32_t input1_size = input1_shape.size();

    if (input0_size > input1_size && input0_size == 5)
    {
        // Reshape input[1] to match input[0] shape
        std::vector<std::uint32_t> uint_vec_target = input0_shape.as_vector();
        std::vector<int> int_vec_target(uint_vec_target.begin(), uint_vec_target.end());

        std::vector<graphlib::OpType::Attr> pos_attrs(int_vec_target.begin(), int_vec_target.end());

        Attrs named_attrs;
        named_attrs["shape"] = int_vec_target;

        tt::graphlib::NodeContext ops1 = dc.op(graphlib::OpType("reshape", pos_attrs, named_attrs), {inputs[1]});

        tt::graphlib::NodeContext result = dc.op(graphlib::OpType("subtract"), {inputs[0], ops1});

        dc.fuse(result, 0);
    }
    else if (input1_size > input0_size && input1_size == 5)
    {
        // Reshape input[0] to match input[1] shape
        std::vector<std::uint32_t> uint_vec_target = input1_shape.as_vector();
        std::vector<int> int_vec_target(uint_vec_target.begin(), uint_vec_target.end());

        std::vector<graphlib::OpType::Attr> pos_attrs(int_vec_target.begin(), int_vec_target.end());

        Attrs named_attrs;
        named_attrs["shape"] = int_vec_target;

        tt::graphlib::NodeContext ops0 = dc.op(graphlib::OpType("reshape", pos_attrs, named_attrs), {inputs[0]});

        tt::graphlib::NodeContext result = dc.op(graphlib::OpType("subtract"), {ops0, inputs[1]});

        dc.fuse(result, 0);
    }
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
    TT_DBG_ASSERT(op.type() == OpType::Subtract, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "subtract::backward should have two input tensors.");
    TT_ASSERT(op.attrs().size() == 0, "subtract::backward should not have any attrs.");
    TT_ASSERT(operand >= 0 && operand < 2, "Invalid operand index.");

    NodeContext op_grad = gradient;
    // For subtract x - y: dx = grad, dy = -grad
    if (operand == 0)
        op_grad = ac.autograd->create_op(ac, graphlib::OpType("nop"), {gradient});
    else
        op_grad = ac.autograd->create_op(
            ac, graphlib::OpType("multiply"), {gradient, ac.autograd->create_constant(ac, -1.0)});

    // Reduce dimensions where broadcasting occurred using reduce_sum
    return op_common::reduce_broadcast_dimensions(ac, op_grad, inputs[operand].shape, gradient.shape);
}

}  // namespace subtract
}  // namespace ops
}  // namespace tt
