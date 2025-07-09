// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "common_utils.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
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

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 2, "OpDivide::eval should have two input tensors.");
    return torch::div(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    return common_utils::compute_elementwise_binary_shape(in_shapes);
}

tt::graphlib::NodeContext backward(
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

    auto input_shape = inputs[operand].shape;
    auto grad_shape = gradient.shape;

    if (input_shape == grad_shape)
    {
        // Shapes match, no broadcasting reduction needed
        return op_grad;
    }

    // Shapes don't match, we need to reduce along broadcast dimensions
    tt::graphlib::NodeContext result_grad = op_grad;
    auto input_dims = input_shape.as_vector();
    auto grad_dims = grad_shape.as_vector();

    // Pad input shape with 1s at the beginning to match gradient rank
    std::vector<std::uint32_t> padded_input_dims = input_dims;
    while (padded_input_dims.size() < grad_dims.size())
    {
        padded_input_dims.insert(padded_input_dims.begin(), 1);
    }

    // Find broadcast dimensions and sum along them, using reduce_sum
    for (size_t i = 0; i < grad_dims.size(); i++)
    {
        if (padded_input_dims[i] < grad_dims[i])
        {
            // Use negative dimension indexing
            int neg_dim = static_cast<int>(i) - static_cast<int>(grad_dims.size());

            Attrs named_attrs = {{"keep_dim", true}, {"dim_arg", std::vector<int>{neg_dim}}};
            result_grad =
                ac.autograd->create_op(ac, graphlib::OpType("reduce_sum", {neg_dim, true}, named_attrs), {result_grad});
        }
    }

    return ac.autograd->create_op(ac, graphlib::OpType("nop", {}, {}), {result_grad});
}

}  // namespace divide

}  // namespace ops

}  // namespace tt
