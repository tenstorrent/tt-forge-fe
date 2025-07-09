// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

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

namespace add
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 2, "OpAdd::eval should have two input tensors.");
    return torch::add(tensors[0], tensors[1]);
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
    TT_ASSERT(inputs.size() == 2, "Add should have two inputs");
    TT_ASSERT(operand >= 0 && operand < 2, "Invalid operand index for add");

    auto input_shape = inputs[operand].shape;
    auto grad_shape = gradient.shape;

    if (input_shape == grad_shape)
    {
        // For addition, gradient flows through unchanged (after handling broadcasting)
        return ac.autograd->create_op(ac, graphlib::OpType("nop", {}, {}), {gradient});
    }

    // Shapes don't match, we need to reduce along broadcast dimensions
    tt::graphlib::NodeContext result_grad = gradient;
    auto input_dims = input_shape.as_vector();
    auto grad_dims = grad_shape.as_vector();

    // Pad input shape with 1s at the beginning to match gradient rank
    std::vector<std::uint32_t> padded_input_dims = input_dims;
    while (padded_input_dims.size() < grad_dims.size())
    {
        padded_input_dims.insert(padded_input_dims.begin(), 1);
    }

    // Find broadcast dimensions and sum along them, using reduce_sum
    for (size_t i = 0; i < padded_input_dims.size(); i++)
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

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_ASSERT(inputs.size() == 2, "Add should have two inputs");

    auto shape_tuple = add::shape(op, inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1L, std::multiplies<long>());
}

}  // namespace add
}  // namespace ops
}  // namespace tt
