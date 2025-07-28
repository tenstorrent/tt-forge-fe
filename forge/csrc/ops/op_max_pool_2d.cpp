// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
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
namespace max_pool_2d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool2d, "Wrong op type.");
    TT_DBG_ASSERT(tensors.size() == 1, "MaxPool2d expects 1 input tensor");

    at::Tensor activations = tensors[0];

    int kernel_height = op.attr_as<int>("kernel_height");
    int kernel_width = op.attr_as<int>("kernel_width");
    int stride_height = op.attr_as<int>("stride_height");
    int stride_width = op.attr_as<int>("stride_width");
    int dilation_height = op.attr_as<int>("dilation_height");
    int dilation_width = op.attr_as<int>("dilation_width");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    int padding_left = op.attr_as<int>("padding_left");
    int padding_right = op.attr_as<int>("padding_right");
    int padding_top = op.attr_as<int>("padding_top");
    int padding_bottom = op.attr_as<int>("padding_bottom");
    bool channel_last = op.attr_as<bool>("channel_last");

    if (channel_last)
    {
        activations = activations.permute({0, 3, 1, 2});
    }

    at::Tensor padded_activations = torch::nn::functional::pad(
        activations,
        torch::nn::functional::PadFuncOptions({padding_left, padding_right, padding_top, padding_bottom})
            .value(-INFINITY));

    at::Tensor result = torch::nn::functional::max_pool2d(
        padded_activations,
        torch::nn::functional::MaxPool2dFuncOptions({kernel_height, kernel_width})
            .stride({stride_height, stride_width})
            .padding(0)
            .dilation({dilation_height, dilation_width})
            .ceil_mode(ceil_mode));

    if (channel_last)
    {
        result = result.permute({0, 2, 3, 1});
    }

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool2d, "Wrong op type.");
    TT_DBG_ASSERT(in_shapes.size() == 1, "MaxPool2d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_DBG_ASSERT(input_shape.size() >= 4, "MaxPool2d input must have at least 4 dimensions");

    int kernel_height = op.attr_as<int>("kernel_height");
    int kernel_width = op.attr_as<int>("kernel_width");
    int stride_height = op.attr_as<int>("stride_height");
    int stride_width = op.attr_as<int>("stride_width");
    int dilation_height = op.attr_as<int>("dilation_height");
    int dilation_width = op.attr_as<int>("dilation_width");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    int padding_left = op.attr_as<int>("padding_left");
    int padding_right = op.attr_as<int>("padding_right");
    int padding_top = op.attr_as<int>("padding_top");
    int padding_bottom = op.attr_as<int>("padding_bottom");
    bool channel_last = op.attr_as<bool>("channel_last");

    uint32_t batch_size = input_shape[0];
    uint32_t channels = channel_last ? input_shape[input_shape.size() - 1] : input_shape[input_shape.size() - 3];
    uint32_t h_in = channel_last ? input_shape[input_shape.size() - 3] : input_shape[input_shape.size() - 2];
    uint32_t w_in = channel_last ? input_shape[input_shape.size() - 2] : input_shape[input_shape.size() - 1];

    int h_numerator = h_in + (padding_top + padding_bottom) - dilation_height * (kernel_height - 1) - 1;
    uint32_t h_out;
    if (ceil_mode)
    {
        h_out = static_cast<uint32_t>(std::ceil(1 + static_cast<float>(h_numerator) / stride_height));
    }
    else
    {
        h_out = static_cast<uint32_t>(std::floor(1 + static_cast<float>(h_numerator) / stride_height));
    }

    int w_numerator = w_in + (padding_left + padding_right) - dilation_width * (kernel_width - 1) - 1;
    uint32_t w_out;
    if (ceil_mode)
    {
        w_out = static_cast<uint32_t>(std::ceil(1 + static_cast<float>(w_numerator) / stride_width));
    }
    else
    {
        w_out = static_cast<uint32_t>(std::floor(1 + static_cast<float>(w_numerator) / stride_width));
    }

    std::vector<uint32_t> output_shape;
    if (channel_last)
    {
        output_shape = {batch_size, h_out, w_out, channels};
    }
    else
    {
        output_shape = {batch_size, channels, h_out, w_out};
    }

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
    TT_DBG_ASSERT(op.type() == OpType::MaxPool2d, "Wrong op type.");
    TT_THROW("OpType::MaxPool2d does not have backward.");
    unreachable();
}

}  // namespace max_pool_2d
}  // namespace ops
}  // namespace tt
