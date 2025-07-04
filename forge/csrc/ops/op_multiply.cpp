// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "autograd/binding.hpp"
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
    TT_DBG_ASSERT(op.type() == OpType::Multiply, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "multiply::shape should have two input shapes.");
    TT_ASSERT(op.attrs().size() == 0, "multiply::shape should not have any attrs.");

    std::vector<graphlib::DimBroadcast> broadcast;
    std::vector<std::uint32_t> output_shape;

    std::vector<std::uint32_t> shape0 = in_shapes[0];
    std::vector<std::uint32_t> shape1 = in_shapes[1];

    // Add leading 1s to the shorter shape
    while (shape0.size() < shape1.size())
    {
        shape0.insert(shape0.begin(), 1);
    }

    while (shape1.size() < shape0.size())
    {
        shape1.insert(shape1.begin(), 1);
    }

    output_shape.resize(shape0.size());

    for (size_t dim = 0; dim < shape0.size(); dim++)
    {
        if (shape0[dim] == shape1[dim])
        {
            output_shape[dim] = shape0[dim];
            continue;
        }

        if (shape1[dim] == 1)
        {
            // Broadcast shape1 to shape0
            int neg_dim = static_cast<int>(dim) - static_cast<int>(shape1.size());
            broadcast.push_back(graphlib::DimBroadcast(1, neg_dim, shape0[dim]));
            output_shape[dim] = shape0[dim];
        }
        else
        {
            TT_ASSERT(
                shape0[dim] == 1,
                "Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to "
                "broadcast");
            // Broadcast shape0 to shape1
            int neg_dim = static_cast<int>(dim) - static_cast<int>(shape0.size());
            broadcast.push_back(graphlib::DimBroadcast(0, neg_dim, shape1[dim]));
            output_shape[dim] = shape1[dim];
        }
    }

    return std::make_tuple(graphlib::Shape::create(output_shape), broadcast);
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
        // Reshape input[1] to match input[0] shape
        std::vector<std::uint32_t> uint_vec_target = input0_shape.as_vector();
        std::vector<int> int_vec_target(uint_vec_target.begin(), uint_vec_target.end());

        std::vector<graphlib::OpType::Attr> pos_attrs(int_vec_target.begin(), int_vec_target.end());

        Attrs named_attrs;
        named_attrs["shape"] = int_vec_target;

        tt::graphlib::NodeContext ops1 = dc.op(graphlib::OpType("reshape", pos_attrs, named_attrs), {inputs[1]});

        tt::graphlib::NodeContext result = dc.op(graphlib::OpType("multiply"), {inputs[0], ops1});

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
    TT_ASSERT(operand < 2, "Invalid operand index.");

    // For multiply x * y: dx = grad * y, dy = grad * x
    NodeContext op_grad = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {gradient, inputs[1 - operand]});

    // If the shapes are the same, no need to reduce dimensions
    const graphlib::Shape &input_shape = inputs[operand].shape;
    const graphlib::Shape &grad_shape = gradient.shape;
    if (input_shape == grad_shape)
        return op_grad;

    // Handle broadcasting - need to reduce dimensions where broadcasting occurred
    std::vector<std::uint32_t> input_dims = input_shape.as_vector();
    std::vector<std::uint32_t> grad_dims = grad_shape.as_vector();

    // Pad shapes with leading 1s to make dimensions match
    std::vector<std::uint32_t> other_dims = inputs[1 - operand].shape.as_vector();

    size_t max_dims = std::max(input_dims.size(), std::max(other_dims.size(), grad_dims.size()));

    std::vector<int> padded_input_dims(max_dims, 1);
    for (size_t i = 0; i < input_dims.size(); i++)
    {
        padded_input_dims[max_dims - input_dims.size() + i] = input_dims[i];
    }

    std::vector<int> padded_grad_dims(max_dims, 1);
    for (size_t i = 0; i < grad_dims.size(); i++)
    {
        padded_grad_dims[max_dims - grad_dims.size() + i] = grad_dims[i];
    }

    // For each dimension, if input_dim < grad_dim, we need to reduce_sum
    for (size_t i = 0; i < max_dims; i++)
    {
        if (padded_input_dims[i] >= padded_grad_dims[i])
            continue;

        int dim = static_cast<int>(i);
        Attrs named_attrs = {{"keep_dim", true}, {"dim_arg", dim}};
        op_grad = ac.autograd->create_op(ac, graphlib::OpType("reduce_sum", {dim, true}, named_attrs), {op_grad});
    }

    return op_grad;
}

}  // namespace multiply
}  // namespace ops
}  // namespace tt
