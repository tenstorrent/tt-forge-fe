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
namespace max_pool_3d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool3d, "Wrong op type.");
    TT_DBG_ASSERT(tensors.size() == 1, "MaxPool3d expects 1 input tensor");

    at::Tensor activations = tensors[0];

    int kernel_depth = op.attr_as<int>("kernel_depth");
    int kernel_height = op.attr_as<int>("kernel_height");
    int kernel_width = op.attr_as<int>("kernel_width");
    int stride_depth = op.attr_as<int>("stride_depth");
    int stride_height = op.attr_as<int>("stride_height");
    int stride_width = op.attr_as<int>("stride_width");
    int dilation = op.attr_as<int>("dilation");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    int padding_left = op.attr_as<int>("padding_left");
    int padding_right = op.attr_as<int>("padding_right");
    int padding_top = op.attr_as<int>("padding_top");
    int padding_bottom = op.attr_as<int>("padding_bottom");
    int padding_front = op.attr_as<int>("padding_front");
    int padding_back = op.attr_as<int>("padding_back");
    bool channel_last = op.attr_as<bool>("channel_last");

    if (channel_last)
    {
        activations = activations.permute({0, 4, 1, 2, 3});
    }

    at::Tensor padded_activations = torch::nn::functional::pad(
        activations,
        torch::nn::functional::PadFuncOptions(
            {padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back})
            .value(-INFINITY));

    at::Tensor result = torch::nn::functional::max_pool3d(
        padded_activations,
        torch::nn::functional::MaxPool3dFuncOptions({kernel_depth, kernel_height, kernel_width})
            .stride({stride_depth, stride_height, stride_width})
            .padding(0)
            .dilation(dilation)
            .ceil_mode(ceil_mode));

    if (channel_last)
    {
        result = result.permute({0, 2, 3, 4, 1});
    }

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool3d, "Wrong op type.");
    TT_DBG_ASSERT(in_shapes.size() == 1, "MaxPool3d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_DBG_ASSERT(input_shape.size() >= 5, "MaxPool3d input must have at least 5 dimensions");

    int kernel_depth = op.attr_as<int>("kernel_depth");
    int kernel_height = op.attr_as<int>("kernel_height");
    int kernel_width = op.attr_as<int>("kernel_width");
    int stride_depth = op.attr_as<int>("stride_depth");
    int stride_height = op.attr_as<int>("stride_height");
    int stride_width = op.attr_as<int>("stride_width");
    int dilation = op.attr_as<int>("dilation");
    int padding_left = op.attr_as<int>("padding_left");
    int padding_right = op.attr_as<int>("padding_right");
    int padding_top = op.attr_as<int>("padding_top");
    int padding_bottom = op.attr_as<int>("padding_bottom");
    int padding_front = op.attr_as<int>("padding_front");
    int padding_back = op.attr_as<int>("padding_back");
    bool channel_last = op.attr_as<bool>("channel_last");

    TT_DBG_ASSERT(dilation == 1, "Currently only support dilation = 1");

    uint32_t batch_size = input_shape[0];
    uint32_t channels, d_in, h_in, w_in;

    if (channel_last)
    {
        d_in = input_shape[1];
        h_in = input_shape[2];
        w_in = input_shape[3];
        channels = input_shape[4];
    }
    else
    {
        channels = input_shape[1];
        d_in = input_shape[2];
        h_in = input_shape[3];
        w_in = input_shape[4];
    }

    uint32_t d_out = (d_in + (padding_front + padding_back) - dilation * (kernel_depth - 1) - 1) / stride_depth + 1;
    uint32_t h_out = (h_in + (padding_top + padding_bottom) - dilation * (kernel_height - 1) - 1) / stride_height + 1;
    uint32_t w_out = (w_in + (padding_left + padding_right) - dilation * (kernel_width - 1) - 1) / stride_width + 1;

    std::vector<uint32_t> output_shape;
    if (channel_last)
    {
        output_shape = {batch_size, d_out, h_out, w_out, channels};
    }
    else
    {
        output_shape = {batch_size, channels, d_out, h_out, w_out};
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
    TT_DBG_ASSERT(op.type() == OpType::MaxPool3d, "Wrong op type.");
    TT_THROW("OpType::MaxPool3d does not have backward.");
    unreachable();
}

}  // namespace max_pool_3d
}  // namespace ops
}  // namespace tt
