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
namespace avg_pool_3d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool3d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "AvgPool3d expects 1 input tensor");

    at::Tensor activations = tensors[0];

    int kernel_depth = op.attr_as<int>("kernel_depth");
    int kernel_height = op.attr_as<int>("kernel_height");
    int kernel_width = op.attr_as<int>("kernel_width");
    int stride_depth = op.attr_as<int>("stride_depth");
    int stride_height = op.attr_as<int>("stride_height");
    int stride_width = op.attr_as<int>("stride_width");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    int padding_left = op.attr_as<int>("padding_left");
    int padding_right = op.attr_as<int>("padding_right");
    int padding_top = op.attr_as<int>("padding_top");
    int padding_bottom = op.attr_as<int>("padding_bottom");
    int padding_front = op.attr_as<int>("padding_front");
    int padding_back = op.attr_as<int>("padding_back");
    bool count_include_pad = op.attr_as<bool>("count_include_pad");
    bool channel_last = op.attr_as<bool>("channel_last");

    TT_ASSERT(
        padding_left == padding_right && padding_top == padding_bottom && padding_front == padding_back,
        "AvgPool3d padding must be symmetric");
    std::vector<int64_t> padding = {padding_left, padding_top, padding_front};

    if (channel_last)
    {
        activations = activations.permute({0, 4, 1, 2, 3});
    }

    at::Tensor result = torch::nn::functional::avg_pool3d(
        activations,
        torch::nn::functional::AvgPool3dFuncOptions({kernel_depth, kernel_height, kernel_width})
            .stride({stride_depth, stride_height, stride_width})
            .padding(padding)
            .ceil_mode(ceil_mode)
            .count_include_pad(count_include_pad));

    if (channel_last)
    {
        result = result.permute({0, 2, 3, 4, 1});
    }

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool3d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "AvgPool3d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() >= 5, "AvgPool3d input must have at least 5 dimensions");

    int kernel_depth = op.attr_as<int>("kernel_depth");
    int kernel_height = op.attr_as<int>("kernel_height");
    int kernel_width = op.attr_as<int>("kernel_width");
    int stride_depth = op.attr_as<int>("stride_depth");
    int stride_height = op.attr_as<int>("stride_height");
    int stride_width = op.attr_as<int>("stride_width");
    int padding_left = op.attr_as<int>("padding_left");
    int padding_right = op.attr_as<int>("padding_right");
    int padding_top = op.attr_as<int>("padding_top");
    int padding_bottom = op.attr_as<int>("padding_bottom");
    int padding_front = op.attr_as<int>("padding_front");
    int padding_back = op.attr_as<int>("padding_back");
    bool channel_last = op.attr_as<bool>("channel_last");

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

    uint32_t d_out = (d_in + (padding_front + padding_back) - kernel_depth) / stride_depth + 1;
    uint32_t h_out = (h_in + (padding_top + padding_bottom) - kernel_height) / stride_height + 1;
    uint32_t w_out = (w_in + (padding_left + padding_right) - kernel_width) / stride_width + 1;

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
    TT_DBG_ASSERT(op.type() == OpType::AvgPool3d, "Wrong op type.");
    TT_THROW("OpType::AvgPool3d does not have backward.");
    unreachable();
}

}  // namespace avg_pool_3d
}  // namespace ops
}  // namespace tt
