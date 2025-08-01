// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace index
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 1, "Index should have one input tensor.");

    const auto &input_tensor = tensors[0];

    int dim = op.attr_as<int>("dim");
    int start = op.attr_as<int>("start");
    int stop = op.attr_as<int>("stop");
    int stride = op.attr_as<int>("stride");

    return input_tensor.slice(dim, start, stop, stride);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_ASSERT(in_shapes.size() == 1, "Index should have one input shape.");

    const auto &input_shape = in_shapes[0];
    std::vector<std::uint32_t> output_shape(input_shape.begin(), input_shape.end());

    int dim = op.attr_as<int>("dim");
    int start = op.attr_as<int>("start");
    int stop = op.attr_as<int>("stop");
    int stride = op.attr_as<int>("stride");

    // Convert positive dim to absolute index
    if (dim >= 0)
    {
        dim -= static_cast<int>(output_shape.size());
    }

    // Convert to positive index
    int abs_dim = dim + static_cast<int>(output_shape.size());
    TT_ASSERT(abs_dim >= 0 && abs_dim < static_cast<int>(output_shape.size()), "Invalid dimension index");

    // Convert start to positive
    if (start < 0)
    {
        start = static_cast<int>(output_shape[abs_dim]) + start;
    }

    // Convert stop to positive
    if (stop < 0)
    {
        stop = static_cast<int>(output_shape[abs_dim]) + stop;
    }

    // Calculate the new dimension size: ceil((stop - start) / stride)
    int new_size = (stop - start + stride - 1) / stride;
    if (new_size < 0)
        new_size = 0;

    output_shape[abs_dim] = static_cast<std::uint32_t>(new_size);

    return std::make_tuple(Shape::create(output_shape), std::vector<DimBroadcast>{});
}

NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_ASSERT(operand == 0, "Index should have exactly one input");

    int dim = op.attr_as<int>("dim");
    int start = op.attr_as<int>("start");
    int stop = op.attr_as<int>("stop");
    int stride = op.attr_as<int>("stride");

    auto input_shape_uint = inputs[0].shape.as_vector();
    std::vector<int64_t> input_shape(input_shape_uint.begin(), input_shape_uint.end());
    int num_dims = static_cast<int>(input_shape.size());

    // Convert to positive dimension
    if (dim < 0)
        dim += num_dims;

    TT_ASSERT(dim >= 0 && dim < num_dims, "Invalid dimension index: " + std::to_string(dim));

    // Convert start to positive
    if (start < 0)
    {
        start = static_cast<int>(input_shape[dim]) + start;
    }

    // Convert stop to positive
    if (stop < 0)
    {
        stop = static_cast<int>(input_shape[dim]) + stop;
    }

    // For each gradient element, create a padded tensor that places it at the correct position
    // Then add it to the result which is initially zero tensor of the same shape as the input
    int num_elements = (stop - start + stride - 1) / stride;  // ceiling division

    // Start with zero tensor
    auto zero_tensor = at::zeros(input_shape, data_format_to_scalar_type(gradient.output_df));
    auto result = ac.autograd->create_constant_tensor(ac, zero_tensor);

    int input_size_dim = static_cast<int>(input_shape[dim]);

    // For each element in the gradient, create a padded version and add it to the result
    for (int i = 0; i < num_elements; i++)
    {
        int target_pos = start + i * stride;

        // Extract the i-th element from gradient
        graphlib::OpType index_op("index");
        index_op.set_attr("dim", dim);
        index_op.set_attr("start", i);
        index_op.set_attr("stop", i + 1);
        index_op.set_attr("stride", 1);

        auto grad_element = ac.autograd->create_op(ac, index_op, {gradient});

        // Calculate padding for the indexed element
        // Use constant_pad with TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]
        std::vector<int> padding(num_dims * 2, 0);

        padding[dim * 2] = target_pos;                           // low padding
        padding[dim * 2 + 1] = input_size_dim - target_pos - 1;  // high padding

        graphlib::OpType constant_pad_op("constant_pad");
        constant_pad_op.set_attr("padding", padding);
        constant_pad_op.set_attr("value", 0.0f);

        auto padded_grad_element = ac.autograd->create_op(ac, constant_pad_op, {grad_element});

        // Add this padded gradient to the result
        graphlib::OpType add_op("add");
        result = ac.autograd->create_op(ac, add_op, {result, padded_grad_element});
    }

    return result;
}

}  // namespace index
}  // namespace ops
}  // namespace tt
