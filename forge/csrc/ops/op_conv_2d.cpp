// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <optional>
#include <vector>

#include "autograd/autograd.hpp"
#include "autograd/binding.hpp"
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
namespace conv_2d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2d, "Wrong op type.");
    TT_ASSERT(tensors.size() <= 3, "Conv ops should have up to three inputs (input, weight, bias)");
    TT_ASSERT(tensors.size() >= 2, "Conv ops should have at least two inputs (input, weight)");

    at::Tensor activations = tensors[0];
    at::Tensor weights = tensors[1];
    std::optional<at::Tensor> bias = tensors.size() == 3 ? std::make_optional(tensors[2]) : std::nullopt;

    // Extract attributes
    std::vector<int> stride = op.attr_as<std::vector<int>>("stride");
    std::vector<int> dilation = op.attr_as<std::vector<int>>("dilation");
    int groups = op.attr_as<int>("groups");
    std::vector<int> padding = op.attr_as<std::vector<int>>("padding");
    bool channel_last = op.attr_as<bool>("channel_last");

    // Convert padding from [pT, pL, pB, pR] to [pL, pR, pT, pB] for torch.nn.functional.pad
    std::vector<int64_t> torch_padding = {padding[1], padding[3], padding[0], padding[2]};
    std::vector<int64_t> torch_stride = {static_cast<int64_t>(stride[0]), static_cast<int64_t>(stride[1])};
    std::vector<int64_t> torch_dilation = {static_cast<int64_t>(dilation[0]), static_cast<int64_t>(dilation[1])};

    if (channel_last)
        activations = activations.permute({0, 3, 1, 2});

    at::Tensor padded_activations = torch::nn::functional::pad(activations, torch_padding);

    // Remember original dtype for result casting
    auto original_dtype = padded_activations.dtype();
    bool cast_result_to_int32 = false;
    bool cast_result_to_original = false;

    if (weights.dtype() == torch::kInt8)
    {
        padded_activations = padded_activations.to(torch::kFloat);
        weights = weights.to(torch::kFloat);

        if (bias.has_value())
            bias = bias->to(torch::kFloat);

        cast_result_to_int32 = true;
    }
    else if (padded_activations.dtype() != weights.dtype())
    {
        // Handle dtype mismatches - cast activations to weights dtype
        padded_activations = padded_activations.to(weights.dtype());
        cast_result_to_original = true;
    }

    // torch requires bias to be in shape (C_out,) so we need to take only channel dimension
    if (bias.has_value())
        bias = bias->view({bias->size(-1)});

    at::Tensor result = at::_ops::conv2d::call(
        padded_activations,
        weights,
        bias,
        c10::fromIntArrayRefSlow(torch_stride),
        c10::fromIntArrayRefSlow({0, 0}),  // padding already applied
        c10::fromIntArrayRefSlow(torch_dilation),
        groups);

    if (channel_last)
        result = result.permute({0, 2, 3, 1});

    if (cast_result_to_int32)
        result = result.to(torch::kInt32);
    else if (cast_result_to_original)
        result = result.to(original_dtype);

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() <= 3, "Conv ops should have up to three inputs (input, weight, bias)");
    TT_ASSERT(in_shapes.size() >= 2, "Conv ops should have at least two inputs (input, weight)");

    const auto &act_shape = in_shapes[0];
    const auto &weight_shape = in_shapes[1];

    std::uint32_t batch_size = act_shape[0];
    std::uint32_t channel_size = weight_shape[0];

    bool channel_last = op.attr_as<bool>("channel_last");
    std::vector<int> padding = op.attr_as<std::vector<int>>("padding");
    std::vector<int> dilation = op.attr_as<std::vector<int>>("dilation");
    std::vector<int> stride = op.attr_as<std::vector<int>>("stride");

    std::uint32_t h_in = channel_last ? act_shape[act_shape.size() - 3] : act_shape[act_shape.size() - 2];
    std::uint32_t w_in = channel_last ? act_shape[act_shape.size() - 2] : act_shape[act_shape.size() - 1];

    std::uint32_t padding_top = padding[0], padding_left = padding[1], padding_bottom = padding[2],
                  padding_right = padding[3];

    std::uint32_t dilation_height = dilation[0], dilation_width = dilation[1];

    std::uint32_t stride_height = stride[0], stride_width = stride[1];

    std::uint32_t h_numerator =
        h_in + (padding_top + padding_bottom) - dilation_height * (weight_shape[weight_shape.size() - 2] - 1) - 1;
    std::uint32_t h_out = static_cast<std::uint32_t>(std::floor(1 + (static_cast<float>(h_numerator) / stride_height)));

    std::uint32_t w_numerator =
        w_in + (padding_left + padding_right) - dilation_width * (weight_shape[weight_shape.size() - 1] - 1) - 1;
    std::uint32_t w_out = static_cast<std::uint32_t>(std::floor(1 + (static_cast<float>(w_numerator) / stride_width)));

    std::vector<std::uint32_t> out_shape;
    if (channel_last)
        out_shape = {batch_size, h_out, w_out, channel_size};
    else
        out_shape = {batch_size, channel_size, h_out, w_out};

    return {Shape::create(out_shape), {}};
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
    TT_DBG_ASSERT(op.type() == OpType::Conv2d, "Wrong op type.");
    TT_ASSERT(false, "Conv2d op does not support backward pass yet.");
    return nullptr;
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2d, "Wrong op type.");
    TT_ASSERT(inputs.size() <= 3, "Conv ops should have up to three inputs (input, weight, bias)");
    TT_ASSERT(inputs.size() >= 2, "Conv ops should have at least two inputs (input, weight)");

    // TTNN can only perform a channel last convolution with its conv2d op.
    // The TTNN conv2d requires the input to be in the shape: (N, H, W, C) or (1, 1, N*H*W, C).
    // It requires the weight to be in the shape: (C_out, C_in, kernel_height, kernel_width).
    // It requires the bias to be in the shape: (1, 1, 1, C_out).
    //
    // If the forge conv2d op is channel-first, we must permute the input (N, C, H, W) tensor to (N, H, W, C)
    // and then transpose it back to (N, C_out, H_out, W_out) afterward.
    //     - This is done with two transposes
    //     - (N, C, H, W) --> transpose(-3, -2): (N, H, C, W) --> transpose(-2, -1): (N, H, W, C)
    // Afterward:
    //     - (N, H_out, W_out, C_out) --> transpose(-2, -1): (N, H_out, C_out, W_out) --> transpose(-3, -2): (N, C_out,
    //     H_out, W_out)

    NodeContext activations = inputs[0];
    NodeContext weight = inputs[1];

    bool has_bias = inputs.size() == 3;
    std::optional<NodeContext> bias = has_bias ? std::make_optional(inputs[2]) : std::nullopt;

    std::vector<int> stride = op.attr_as<std::vector<int>>("stride");
    std::vector<int> dilation = op.attr_as<std::vector<int>>("dilation");
    std::vector<int> padding = op.attr_as<std::vector<int>>("padding");
    int groups = op.attr_as<int>("groups");
    bool channel_last = op.attr_as<bool>("channel_last");

    std::size_t bias_shape_size = has_bias ? bias->shape.as_vector().size() : 0;
    std::size_t activations_shape_size = activations.shape.as_vector().size();

    bool is_channel_last = channel_last;

    // Unsqueeze bias to match the number of dimensions of the activations
    bool is_bias_unchanged = true;
    if (has_bias && bias_shape_size < activations_shape_size)
    {
        while (bias_shape_size < activations_shape_size)
        {
            *bias = dc.op(graphlib::OpType("unsqueeze", {}, {{"dim", 0}}), {*bias});
            bias_shape_size++;
        }
        is_bias_unchanged = false;
    }

    // Convert to channel-last if needed
    if (!is_channel_last)
    {
        activations = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {activations});
        activations = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {activations});
    }

    // Check if bias needs transpose for channel alignment:
    // weight shape: [C_out, C_in, kernel_h, kernel_w], bias shape: [..., C_out]
    // Ensure bias[-1] matches weight[0] (output channels)
    if (has_bias && bias->shape.as_vector()[bias_shape_size - 1] != weight.shape.as_vector()[0] && !is_channel_last)
    {
        *bias = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {*bias});
        *bias = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {*bias});

        is_bias_unchanged = false;
    }

    // Only want to re-create the Conv2d op if something has changed. Otherwise the compiler will infinitely
    // decompose the same Conv2d over and over.
    if (!is_bias_unchanged || !is_channel_last)
    {
        graphlib::OpType::Attrs conv_attrs = {
            {"stride", stride},
            {"dilation", dilation},
            {"groups", groups},
            {"padding", padding},
            {"channel_last", true}};

        std::vector<NodeContext> new_inputs = {activations, weight};
        if (has_bias)
            new_inputs.push_back(*bias);

        NodeContext result = dc.op(graphlib::OpType("conv2d", {}, conv_attrs), new_inputs);

        // Convert back to channel-first if needed
        if (!is_channel_last)
        {
            result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {result});
            result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {result});
        }

        dc.fuse(result);
    }
}

}  // namespace conv_2d
}  // namespace ops
}  // namespace tt
