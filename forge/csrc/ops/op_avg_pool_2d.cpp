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
namespace avg_pool_2d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool2d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "AvgPool2d expects 1 input tensor");

    at::Tensor activations = tensors[0];

    auto kernel = op.attr_as<std::vector<int>>("kernel");
    TT_ASSERT(kernel.size() == 2, "kernel array must have 2 elements [kH, kW]");
    int kernel_height = kernel[0];
    int kernel_width = kernel[1];

    auto stride = op.attr_as<std::vector<int>>("stride");
    TT_ASSERT(stride.size() == 2, "stride array must have 2 elements [sH, sW]");
    int stride_height = stride[0];
    int stride_width = stride[1];

    auto padding = op.attr_as<std::vector<int>>("padding");
    TT_ASSERT(padding.size() == 4, "padding array must have 4 elements [pT, pL, pB, pR]");
    int padding_top = padding[0];
    int padding_left = padding[1];
    int padding_bottom = padding[2];
    int padding_right = padding[3];

    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    bool count_include_pad = op.attr_as<bool>("count_include_pad");
    bool channel_last = op.attr_as<bool>("channel_last");

    TT_ASSERT(padding_left == padding_right && padding_top == padding_bottom, "AvgPool2d padding must be symmetric");

    if (channel_last)
    {
        activations = activations.permute({0, 3, 1, 2});
    }

    at::Tensor result = torch::nn::functional::avg_pool2d(
        activations,
        torch::nn::functional::AvgPool2dFuncOptions({kernel_height, kernel_width})
            .stride({stride_height, stride_width})
            .padding({padding_left, padding_top})
            .ceil_mode(ceil_mode)
            .count_include_pad(count_include_pad));

    if (channel_last)
    {
        result = result.permute({0, 2, 3, 1});
    }

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool2d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "AvgPool2d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() >= 4, "AvgPool2d input must have at least 4 dimensions");

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

    bool channel_last = op.attr_as<bool>("channel_last");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");

    TT_ASSERT(dilation_height == 1 && dilation_width == 1, "Currently only support dilation = 1");
    TT_ASSERT(padding_left == padding_right && padding_top == padding_bottom, "AvgPool2d padding must be symmetric");

    uint32_t batch_size = input_shape[0];
    uint32_t channels = channel_last ? input_shape[input_shape.size() - 1] : input_shape[input_shape.size() - 3];
    uint32_t h_in = channel_last ? input_shape[input_shape.size() - 3] : input_shape[input_shape.size() - 2];
    uint32_t w_in = channel_last ? input_shape[input_shape.size() - 2] : input_shape[input_shape.size() - 1];

    uint32_t h_out;
    uint32_t w_out;
    if (ceil_mode)
    {
        h_out = static_cast<uint32_t>(std::ceil(
            static_cast<float>(h_in + 2 * padding_top - dilation_height * (kernel_height - 1) - 1) / stride_height +
            1));
        w_out = static_cast<uint32_t>(std::ceil(
            static_cast<float>(w_in + 2 * padding_left - dilation_width * (kernel_width - 1) - 1) / stride_width + 1));
    }
    else
    {
        h_out = static_cast<uint32_t>(std::floor(
            static_cast<float>(h_in + 2 * padding_top - dilation_height * (kernel_height - 1) - 1) / stride_height +
            1));
        w_out = static_cast<uint32_t>(std::floor(
            static_cast<float>(w_in + 2 * padding_left - dilation_width * (kernel_width - 1) - 1) / stride_width + 1));
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
    TT_DBG_ASSERT(op.type() == OpType::AvgPool2d, "Wrong op type.");
    TT_THROW("OpType::AvgPool2d does not have backward.");
    unreachable();
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool2d, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "AvgPool2d expects 1 input");

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
    bool count_include_pad = op.attr_as<bool>("count_include_pad");

    TT_ASSERT(dilation_height == 1 && dilation_width == 1, "Currently only support dilation = 1");

    NodeContext activations = inputs[0];
    Shape activations_shape = activations.shape;

    uint32_t w, cin, y, x;
    if (channel_last)
    {
        w = activations_shape[0];
        y = activations_shape[activations_shape.size() - 3];
        x = activations_shape[activations_shape.size() - 2];
        cin = activations_shape[activations_shape.size() - 1];
    }
    else
    {
        w = activations_shape[0];
        cin = activations_shape[activations_shape.size() - 3];
        y = activations_shape[activations_shape.size() - 2];
        x = activations_shape[activations_shape.size() - 1];
    }
    std::vector<int> original_padding = padding;

    // Handle ceil_mode
    if (ceil_mode)
    {
        // Calculate padding adjustments for ceil mode
        auto calculate_ceil_pad =
            [](int original_dim, int kernel_dim, int stride_dim, int pad_prefix, int pad_suffix) -> int
        {
            if (pad_suffix >= kernel_dim)
                return 0;

            int out_dim_ceil = static_cast<int>(
                std::ceil(static_cast<float>(original_dim + pad_prefix + pad_suffix - kernel_dim) / stride_dim + 1));
            int total_padding = stride_dim * (out_dim_ceil - 1) - original_dim + kernel_dim;
            int padding_to_add = total_padding - pad_prefix - pad_suffix;
            return std::max(0, padding_to_add);
        };

        int ceil_pad_right = calculate_ceil_pad(x, kernel_width, stride_width, padding_left, padding_right);
        int ceil_pad_bottom = calculate_ceil_pad(y, kernel_height, stride_height, padding_top, padding_bottom);

        if (ceil_pad_right == 0 && ceil_pad_bottom == 0)
        {
            ceil_mode = false;
        }
        else
        {
            padding[1] += ceil_pad_right;   // padding_right
            padding[3] += ceil_pad_bottom;  // padding_bottom
        }
    }

    int kH = kernel_height;
    int kW = kernel_width;

    // Check for global average pooling
    bool is_global_avg =
        (static_cast<int>(y) == kH && static_cast<int>(x) == kW &&
         ((stride_height == kH && stride_width == kW) ||
          (padding[0] == 0 && padding[1] == 0 && padding[2] == 0 && padding[3] == 0)));

    if (is_global_avg)
    {
        NodeContext result = activations;  // Initialize with input
        if (channel_last)
        {
            result = dc.op(
                graphlib::OpType(
                    "reshape",
                    {{"shape",
                      std::vector<int>{static_cast<int>(w), 1, static_cast<int>(y * x), static_cast<int>(cin)}}}),
                {activations});
            result = dc.op(
                graphlib::OpType("reduce_avg", {{"dim_arg", std::vector<int>{-2}}, {"keep_dim", true}}), {result});
            result = dc.op(
                graphlib::OpType(
                    "reshape", {{"shape", std::vector<int>{static_cast<int>(w), 1, 1, static_cast<int>(cin)}}}),
                {result});
        }
        else
        {
            result = dc.op(
                graphlib::OpType(
                    "reshape",
                    {{"shape",
                      std::vector<int>{static_cast<int>(w), 1, static_cast<int>(cin), static_cast<int>(y * x)}}}),
                {activations});
            result = dc.op(graphlib::OpType("transpose", {{"dim0", 2}, {"dim1", 3}}), {result});
            result = dc.op(
                graphlib::OpType("reduce_avg", {{"dim_arg", std::vector<int>{-2}}, {"keep_dim", true}}), {result});
            result = dc.op(graphlib::OpType("transpose", {{"dim0", 2}, {"dim1", 3}}), {result});
            result = dc.op(
                graphlib::OpType(
                    "reshape", {{"shape", std::vector<int>{static_cast<int>(w), static_cast<int>(cin), 1, 1}}}),
                {result});
        }
        dc.fuse(result);
        return;
    }

    // Create weight tensor for convolution
    float weight_value = 1.0f / (kH * kW);
    at::Tensor weight_tensor = weight_value * torch::ones({static_cast<int64_t>(cin), 1, kH, kW});
    NodeContext weight = dc.tensor(weight_tensor);

    // Perform convolution
    NodeContext result = dc.op(
        graphlib::OpType(
            "conv2d",
            {{"stride", std::vector<int>{stride_height, stride_width}},
             {"dilation", std::vector<int>{dilation_height, dilation_width}},
             {"groups", static_cast<int>(cin)},
             {"padding", padding},
             {"channel_last", channel_last}}),
        {activations, weight});

    // Handle count_include_pad=False or ceil_mode padding correction
    bool need_padding_correction =
        (padding[0] != 0 || padding[1] != 0 || padding[2] != 0 || padding[3] != 0) && (ceil_mode || !count_include_pad);

    if (need_padding_correction)
    {
        Shape result_shape = result.shape;
        int y_out, x_out;

        if (channel_last)
        {
            y_out = result_shape[result_shape.size() - 3];
            x_out = result_shape[result_shape.size() - 2];

            result = dc.op(
                graphlib::OpType(
                    "reshape",
                    {{"shape",
                      std::vector<int>{
                          static_cast<int>(w), 1, static_cast<int>(y_out * x_out), static_cast<int>(cin)}}}),
                {result});
        }
        else
        {
            y_out = result_shape[result_shape.size() - 2];
            x_out = result_shape[result_shape.size() - 1];

            result = dc.op(
                graphlib::OpType(
                    "reshape",
                    {{"shape",
                      std::vector<int>{
                          static_cast<int>(w), 1, static_cast<int>(cin), static_cast<int>(y_out * x_out)}}}),
                {result});
            result = dc.op(graphlib::OpType("transpose", {{"dim0", 2}, {"dim1", 3}}), {result});
        }

        // Create padding correction matrix
        int corrected_y = y;
        int corrected_x = x;
        std::vector<int> correction_padding = padding;

        if (!count_include_pad)
        {
            // For count_include_pad=False, correct all padding
            corrected_y = y;
            corrected_x = x;
            correction_padding = padding;
        }
        else if (ceil_mode)
        {
            // For ceil_mode=True, only correct ceil padding
            corrected_y = y + original_padding[2] + original_padding[3];
            corrected_x = x + original_padding[0] + original_padding[1];
            int ceil_pad_right = padding[1] - original_padding[1];
            int ceil_pad_bottom = padding[3] - original_padding[3];
            correction_padding = {0, ceil_pad_right, 0, ceil_pad_bottom};
        }

        // Create correction matrix (simplified implementation)
        at::Tensor padding_mask =
            torch::zeros({static_cast<int64_t>(corrected_y), static_cast<int64_t>(corrected_x)}, torch::kUInt8)
                .view({1, 1, static_cast<int64_t>(corrected_y), static_cast<int64_t>(corrected_x)});
        padding_mask = torch::nn::functional::pad(
            padding_mask,
            torch::nn::functional::PadFuncOptions(
                {correction_padding[0], correction_padding[1], correction_padding[2], correction_padding[3]})
                .value(1));

        at::Tensor correction_weights = torch::ones({1, 1, kH, kW}, torch::kUInt8);
        at::Tensor picker = torch::nn::functional::conv2d(
            padding_mask,
            correction_weights,
            torch::nn::functional::Conv2dFuncOptions().stride({stride_height, stride_width}));
        picker = picker.squeeze(0).squeeze(0);

        // Calculate correction factors
        at::Tensor onehot = picker.to(torch::kBool).to(torch::kUInt8);
        int kernel_volume = kH * kW;
        picker = onehot * kernel_volume / (kernel_volume - picker);
        picker = picker + 1 - onehot;

        picker = picker.reshape(-1);
        at::Tensor rows_cols = torch::arange(picker.size(0));

        int picker_dim = static_cast<int>(picker.size(0));
        at::Tensor correction_matrix = torch::sparse_coo_tensor(
                                           torch::stack({rows_cols, rows_cols}),
                                           picker.to(torch::kFloat32),
                                           {picker_dim, picker_dim},
                                           torch::kFloat32)
                                           .coalesce();

        NodeContext correction_tensor = dc.tensor(correction_matrix.to_dense());
        result = dc.op(graphlib::OpType("matmul"), {correction_tensor, result});

        if (channel_last)
        {
            result = dc.op(
                graphlib::OpType(
                    "reshape",
                    {{"shape",
                      std::vector<int>{
                          static_cast<int>(w),
                          static_cast<int>(y_out),
                          static_cast<int>(x_out),
                          static_cast<int>(cin)}}}),
                {result});
        }
        else
        {
            result = dc.op(graphlib::OpType("transpose", {{"dim0", 2}, {"dim1", 3}}), {result});
            result = dc.op(
                graphlib::OpType(
                    "reshape",
                    {{"shape",
                      std::vector<int>{
                          static_cast<int>(w),
                          static_cast<int>(cin),
                          static_cast<int>(y_out),
                          static_cast<int>(x_out)}}}),
                {result});
        }
    }

    dc.fuse(result);

    // TODO: After fix to the TTNN avg_pool2d op, we can use this code

    // TTNN can only perform a channel last pooling with its avg_pool2d op.
    // The TTNN avg_pool2d requires the input to be in the shape: (N, H, W, C) or (1, 1, N*H*W, C).
    // If the forge avg_pool2d op is channel-first, we must permute the input (N, C, H, W) tensor to (N, H, W, C)
    // and then transpose it back to (N, H_out, W_out, C_out) afterward.
    //     - This is done with two transposes
    //     - (N, C, H, W) --> transpose(-3, -2): (N, H, C, W) --> transpose(-2, -1): (N, H, W, C)
    // Afterward:
    //     - (N, H_out, W_out, C_out) --> transpose(-2, -1): (N, H_out, C_out, W_out) --> transpose(-3, -2): (N, C_out,
    //     H_out, W_out)

    // bool channel_last = op.attr_as<bool>("channel_last");
    // if (channel_last)
    //     return;

    // NodeContext activations = inputs[0];
    // activations = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {activations});
    // activations = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {activations});
    // NodeContext result = dc.op(
    //     graphlib::OpType(
    //         "avg_pool2d",
    //         {},
    //         {{"kernel_height", kernel_height},
    //          {"kernel_width", kernel_width},
    //          {"stride_height", stride_height},
    //          {"stride_width", stride_width},
    //          {"padding_left", padding_left},
    //          {"padding_right", padding_right},
    //          {"padding_top", padding_top},
    //          {"padding_bottom", padding_bottom},
    //          {"channel_last", true},
    //          {"ceil_mode", ceil_mode},
    //          {"dilation_height", dilation_height},
    //          {"dilation_width", dilation_width},
    //          {"count_include_pad", count_include_pad}}),
    //     {activations});
    // result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {result});
    // result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {result});
    // dc.fuse(result);
    // return;
}

}  // namespace avg_pool_2d
}  // namespace ops
}  // namespace tt
