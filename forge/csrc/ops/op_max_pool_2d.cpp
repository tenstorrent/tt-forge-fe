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

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool2d, "Wrong op type.");
    TT_DBG_ASSERT(tensors.size() == 1, "MaxPool2d expects 1 input tensor");

    at::Tensor activations = tensors[0];

    auto kernel = op.attr_as<std::vector<int>>("kernel");
    TT_ASSERT(kernel.size() == 2, "kernel array must have 2 elements [kH, kW]");
    int kernel_height = kernel[0];
    int kernel_width = kernel[1];

    auto stride = op.attr_as<std::vector<int>>("stride");
    TT_ASSERT(stride.size() == 2, "stride array must have 2 elements [sH, sW]");
    int stride_height = stride[0];
    int stride_width = stride[1];

    auto dilation = op.attr_as<std::vector<int>>("dilation");
    TT_ASSERT(dilation.size() == 2, "dilation array must have 2 elements [dH, dW]");
    int dilation_height = dilation[0];
    int dilation_width = dilation[1];

    auto padding = op.attr_as<std::vector<int>>("padding");
    TT_ASSERT(padding.size() == 4, "padding array must have 4 elements [pT, pL, pB, pR]");
    int padding_top = padding[0];
    int padding_left = padding[1];
    int padding_bottom = padding[2];
    int padding_right = padding[3];

    bool ceil_mode = op.attr_as<bool>("ceil_mode");
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
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool2d, "Wrong op type.");
    TT_DBG_ASSERT(in_shapes.size() == 1, "MaxPool2d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_DBG_ASSERT(input_shape.size() >= 4, "MaxPool2d input must have at least 4 dimensions");

    auto kernel = op.attr_as<std::vector<int>>("kernel");
    TT_ASSERT(kernel.size() == 2, "kernel array must have 2 elements [kH, kW]");
    int kernel_height = kernel[0];
    int kernel_width = kernel[1];

    auto stride = op.attr_as<std::vector<int>>("stride");
    TT_ASSERT(stride.size() == 2, "stride array must have 2 elements [sH, sW]");
    int stride_height = stride[0];
    int stride_width = stride[1];

    auto dilation = op.attr_as<std::vector<int>>("dilation");
    TT_ASSERT(dilation.size() == 2, "dilation array must have 2 elements [dH, dW]");
    int dilation_height = dilation[0];
    int dilation_width = dilation[1];

    auto padding = op.attr_as<std::vector<int>>("padding");
    TT_ASSERT(padding.size() == 4, "padding array must have 4 elements [pT, pL, pB, pR]");
    int padding_top = padding[0];
    int padding_left = padding[1];
    int padding_bottom = padding[2];
    int padding_right = padding[3];

    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    bool channel_last = op.attr_as<bool>("channel_last");

    uint32_t batch_size = input_shape[0];
    uint32_t channels = channel_last ? input_shape[input_shape.size() - 1] : input_shape[input_shape.size() - 3];
    uint32_t h_in = channel_last ? input_shape[input_shape.size() - 3] : input_shape[input_shape.size() - 2];
    uint32_t w_in = channel_last ? input_shape[input_shape.size() - 2] : input_shape[input_shape.size() - 1];

    TT_ASSERT(dilation_height == 1 && dilation_width == 1, "Currently only support dilation = 1");
    TT_ASSERT(padding_left == padding_right && padding_top == padding_bottom, "MaxPool2d padding must be symmetric");

    uint32_t h_out;
    uint32_t w_out;
    if (ceil_mode)
    {
        h_out = static_cast<uint32_t>(std::ceil(
            1 +
            static_cast<float>(h_in + 2 * padding_top - dilation_height * (kernel_height - 1) - 1) / stride_height));
        w_out = static_cast<uint32_t>(std::ceil(
            1 + static_cast<float>(w_in + 2 * padding_left - dilation_width * (kernel_width - 1) - 1) / stride_width));
    }
    else
    {
        h_out = static_cast<uint32_t>(std::floor(
            1 +
            static_cast<float>(h_in + 2 * padding_top - dilation_height * (kernel_height - 1) - 1) / stride_height));
        w_out = static_cast<uint32_t>(std::floor(
            1 + static_cast<float>(w_in + 2 * padding_left - dilation_width * (kernel_width - 1) - 1) / stride_width));
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

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool2d, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "MaxPool2d expects 1 input");

    bool channel_last = op.attr_as<bool>("channel_last");
    NodeContext activations = inputs[0];

    // TTNN can only perform a channel last pooling with its maxpool2d op.
    // The TTNN maxpool2d requires the input to be in the shape: (N, H, W, C).
    // If the forge maxpool2d op is not channel last, we must permute the input (N, C, H, W) tensor to (N, H, W, C)
    // and then transpose it back to (N, C_out, H_out, W_out) afterward.

    if (channel_last)
        return;

    // (N, C, H, W) --> transpose(-3, -2): (N, H, C, W) --> transpose(-2, -1): (N, H, W, C)
    activations = dc.op(Op(OpType::Transpose, {{"dim0", -3}, {"dim1", -2}}), {activations});
    activations = dc.op(Op(OpType::Transpose, {{"dim0", -2}, {"dim1", -1}}), {activations});

    // Create a new MaxPool2d operation with channel_last=true
    NodeContext result = dc.op(
        Op(OpType::MaxPool2d,
           {{"kernel", op.attr_as<std::vector<int>>("kernel")},
            {"stride", op.attr_as<std::vector<int>>("stride")},
            {"dilation", op.attr_as<std::vector<int>>("dilation")},
            {"padding", op.attr_as<std::vector<int>>("padding")},
            {"ceil_mode", op.attr_as<bool>("ceil_mode")},
            {"channel_last", true}}),
        {activations});

    // Transpose back to channel first: (N, H_out, W_out, C_out) --> transpose(-2, -1): (N, H_out, C_out, W_out) -->
    // transpose(-3, -2): (N, C_out, H_out, W_out)
    result = dc.op(Op(OpType::Transpose, {{"dim0", -2}, {"dim1", -1}}), {result});
    result = dc.op(Op(OpType::Transpose, {{"dim0", -3}, {"dim1", -2}}), {result});

    dc.fuse(result);
    return;
}

}  // namespace max_pool_2d
}  // namespace ops
}  // namespace tt
