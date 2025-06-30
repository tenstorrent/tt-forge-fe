// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
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
at::Tensor Op::multiply_eval(const std::vector<at::Tensor> &tensors) const
{
    TT_ASSERT(tensors.size() == 2, "OpMultiply::eval should have two input tensors.");
    return torch::multiply(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::multiply_shape(
    const std::vector<std::vector<std::uint32_t>> &in_shapes) const
{
    TT_ASSERT(in_shapes.size() == 2, "OpMultiply::shape should have two input shapes.");

    std::vector<graphlib::DimBroadcast> broadcast;
    std::vector<std::uint32_t> output_shape;

    // Create copies of input shapes that we can modify
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

    // Now both shapes are the same length
    output_shape.resize(shape0.size());

    for (size_t dim = 0; dim < shape0.size(); dim++)
    {
        if (shape0[dim] != shape1[dim])
        {
            if (shape1[dim] == 1)
            {
                // Broadcast shape1 to shape0
                int neg_index = static_cast<int>(dim) - static_cast<int>(shape1.size());
                broadcast.push_back(graphlib::DimBroadcast(1, neg_index, shape0[dim]));
                output_shape[dim] = shape0[dim];
            }
            else
            {
                TT_ASSERT(
                    shape0[dim] == 1,
                    "Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to "
                    "broadcast");
                // Broadcast shape0 to shape1
                int neg_index = static_cast<int>(dim) - static_cast<int>(shape0.size());
                broadcast.push_back(graphlib::DimBroadcast(0, neg_index, shape1[dim]));
                output_shape[dim] = shape1[dim];
            }
        }
        else
        {
            output_shape[dim] = shape0[dim];
        }
    }

    return std::make_tuple(graphlib::Shape::create(output_shape), broadcast);
}

tt::graphlib::NodeContext Op::multiply_backward(
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient) const
{
    TT_ASSERT(inputs.size() == 2, "OpMultiply::backward should have two input tensors.");
    TT_ASSERT(operand < 2, "Invalid operand index.");

    // For multiply x * y: dx = grad * y, dy = grad * x
    NodeContext op_grad = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {gradient, inputs[1 - operand]});

    // Handle broadcasting - need to reduce dimensions where broadcasting occurred
    std::vector<std::uint32_t> input_dims = inputs[operand].shape.as_vector();
    std::vector<std::uint32_t> grad_dims = gradient.shape.as_vector();

    // If shapes are different, we need to handle potential broadcasting
    if (input_dims != grad_dims)
    {
        std::vector<std::uint32_t> other_dims = inputs[1 - operand].shape.as_vector();

        // Pad shapes with leading 1s to make dimensions match
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

        // For each dimension, if input dim < grad dim, we need to reduce_sum
        for (size_t i = 0; i < max_dims; i++)
        {
            if (padded_input_dims[i] < padded_grad_dims[i])
            {
                // Calculate negative indexing as in Python
                int neg_index = static_cast<int>(i) - static_cast<int>(max_dims);

                // Create attribute dict for reduce_sum
                graphlib::AttributeMap reduce_attrs;
                reduce_attrs["keep_dim"] = true;
                std::vector<int> dim_arg = {neg_index};
                reduce_attrs["dim_arg"] = dim_arg;

                // Apply reduce_sum operation
                op_grad = ac.autograd->create_op(ac, graphlib::OpType("reduce_sum"), {op_grad}, reduce_attrs);
            }
        }
    }

    return op_grad;
}

long Op::multiply_initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    auto shape_tuple = multiply_shape(inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace ops
}  // namespace tt
