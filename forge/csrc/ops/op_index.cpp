// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
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

using op_common::decompose_constant_mode;
using op_common::PaddingParams;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
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
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_ASSERT(in_shapes.size() == 1, "Index should have one input shape.");

    const auto &input_shape = in_shapes[0];
    std::vector<std::uint32_t> output_shape(input_shape.begin(), input_shape.end());

    int dim = op.attr_as<int>("dim");
    int start = op.attr_as<int>("start");
    int stop = op.attr_as<int>("stop");
    int stride = op.attr_as<int>("stride");

    // Convert dim to positive
    if (dim < 0)
        dim += static_cast<int>(output_shape.size());

    TT_ASSERT(dim >= 0 && dim < static_cast<int>(output_shape.size()), "Invalid dimension index");

    // Handle stride=0 case (use full dimension size as stride)
    if (stride == 0)
        stride = static_cast<int>(output_shape[dim]);

    // Convert start to positive
    if (start < 0)
        start = static_cast<int>(output_shape[dim]) + start;

    // Convert stop to positive
    if (stop < 0)
        stop = static_cast<int>(output_shape[dim]) + stop;

    // Calculate the new dimension size: ceil((stop - start) / stride)
    int new_size = (stop - start + stride - 1) / stride;
    if (new_size < 0)
        new_size = 0;

    output_shape[dim] = static_cast<std::uint32_t>(new_size);

    return std::make_tuple(Shape::create(output_shape), std::vector<DimBroadcast>{});
}

NodeContext backward(

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

    TT_ASSERT(stride == 1, "Only stride == 1 is supported for index op backward");

    auto input_shape_uint = inputs[0].shape.as_vector();
    std::vector<int64_t> input_shape(input_shape_uint.begin(), input_shape_uint.end());
    int num_dims = static_cast<int>(input_shape.size());

    // Convert to positive dimension
    if (dim < 0)
        dim += num_dims;

    TT_ASSERT(dim >= 0 && dim < num_dims, "Invalid dimension index: " + std::to_string(dim));

    // Handle stride=0 case (use full dimension size as stride)
    if (stride == 0)
        stride = static_cast<int>(input_shape[dim]);

    // Convert start to positive
    if (start < 0)
        start = static_cast<int>(input_shape[dim]) + start;

    // Convert stop to positive
    if (stop < 0)
        stop = static_cast<int>(input_shape[dim]) + stop;

    // Calculate padding needed to restore the original size
    int left_pad = start;
    int right_pad = static_cast<int>(input_shape[dim]) - stop;
    int shape_size = static_cast<int>(gradient.shape.size());

    // Use the simplified constructor that pads on the specific dimension
    PaddingParams params(dim, left_pad, right_pad, shape_size);

    return decompose_constant_mode(ac, gradient, params, 0.0f);
}

}  // namespace index
}  // namespace ops
}  // namespace tt
