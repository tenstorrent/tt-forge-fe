// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "lower_to_forge/common.hpp"
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
namespace pad
{
using namespace graphlib;

NodeContext create_constant_pad_op(
    DecomposingContext &dc,
    const NodeContext &input,
    const std::vector<int> &padding,
    int height_dim,
    int width_dim,
    float value);

NodeContext create_pad(DecomposingContext &dc, const std::vector<int> &shape, float value, DataFormat data_format);

NodeContext extract(DecomposingContext &dc, const NodeContext &input, int dim_axis, int start, int stop);

NodeContext repeat_vector(DecomposingContext &dc, const NodeContext &input, int n_repeats, int axis);

NodeContext concat_patches(
    DecomposingContext &dc,
    const NodeContext *first_patch,
    const NodeContext &center,
    const NodeContext *second_patch,
    int dim_axis);

void decompose_replicate_mode(
    DecomposingContext &dc,
    const NodeContext &activations,
    const std::vector<int> &padding,
    int height_dim,
    int width_dim);

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Pad, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Pad should have exactly 1 input tensor");
    TT_ASSERT(op.attrs().size() == 4, "Pad should have 4 attributes: padding, mode, value, channel_last");

    auto padding = op.attr_as<std::vector<int>>("padding");
    auto mode = op.attr_as<int>("mode");
    auto value = op.attr_as<float>("value");
    auto channel_last = op.attr_as<bool>("channel_last");

    at::Tensor input_tensor = tensors[0];

    std::vector<std::string> mode_options = {"constant", "replicate", "reflect"};
    TT_ASSERT(mode >= 0 && mode < 3, "Invalid padding mode");

    if (!channel_last)
    {
        return at::pad(input_tensor, std::vector<int64_t>(padding.begin(), padding.end()), mode_options[mode], value);
    }

    // When channel_last=True, the input tensor is already in format (N, D1, D2, ..., Dn, C)
    // We need to move the channel from the last position to position 1
    // to get (N, C, D1, D2, ..., Dn) which is what PyTorch expects for padding
    int ndim = input_tensor.dim();

    // Create permutation that moves channel from the last dim to position 1
    // For a tensor (N, D1, D2, ..., Dn, C) -> (N, C, D1, D2, ..., Dn)
    std::vector<int64_t> perm = {0, ndim - 1};                 // batch, channel
    for (int64_t i = 1; i < ndim - 1; i++) perm.push_back(i);  // spatial dimensions: 1, 2, ..., ndim-2

    at::Tensor transposed = input_tensor.permute(perm);

    at::Tensor padded =
        at::pad(transposed, std::vector<int64_t>(padding.begin(), padding.end()), mode_options[mode], value);

    // For a tensor (N, C, D1, D2, ..., Dn) -> (N, D1, D2, ..., Dn, C)
    std::vector<int64_t> reverse_perm = {0};                       // batch
    for (int64_t i = 2; i < ndim; i++) reverse_perm.push_back(i);  // spatial dimensions: 2, 3, ..., ndim-1
    reverse_perm.push_back(1);                                     // channel (move from position 1 to end)

    return padded.permute(reverse_perm);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Pad, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Pad should have exactly 1 input shape");
    TT_ASSERT(op.attrs().size() == 4, "Pad should have 4 attributes: padding, mode, value, channel_last");

    auto padding = op.attr_as<std::vector<int>>("padding");
    auto channel_last = op.attr_as<bool>("channel_last");

    auto shape = in_shapes[0];

    if (channel_last)
    {
        // For channel_last format, padding affects the spatial dimensions
        // padding format: [left, right] or [left, right, top, bottom]
        shape[shape.size() - 2] += padding[0] + padding[1];  // width padding
        if (padding.size() == 4)
            shape[shape.size() - 3] += padding[2] + padding[3];  // height padding
    }
    else
    {
        // For channel_first format, padding affects the last dimensions
        shape[shape.size() - 1] += padding[0] + padding[1];  // width padding
        if (padding.size() == 4)
            shape[shape.size() - 2] += padding[2] + padding[3];  // height padding
    }

    return std::make_tuple(Shape::create(shape), std::vector<DimBroadcast>{});
}

// currently this implementation is not used since we decompose pad so when we run backward pass it's run on the
// decomposed ops (but if we introduce direct mapping to TTIR then we can use this implementation)
NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    // Backward pass for pad is to remove the padding (narrow operation)
    TT_DBG_ASSERT(op.type() == OpType::Pad, "Wrong op type.");
    TT_ASSERT(operand == 0, "Pad should have exactly 1 operand");
    TT_ASSERT(op.attrs().size() == 4, "Pad should have 4 attributes: padding, mode, value, channel_last");

    auto padding = op.attr_as<std::vector<int>>("padding");
    auto channel_last = op.attr_as<bool>("channel_last");

    TT_ASSERT(padding.size() == 2 || padding.size() == 4, "Not supported padding type");

    int shape_size = static_cast<int>(gradient.shape.size());
    int height_dim, width_dim;
    if (channel_last)
    {
        height_dim = shape_size - 3;  // channel_last: height is at -3
        width_dim = shape_size - 2;   // channel_last: width is at -2
    }
    else
    {
        height_dim = shape_size - 2;  // channel_first: height is at -2
        width_dim = shape_size - 1;   // channel_first: width is at -1
    }

    int original_height = gradient.shape[height_dim];
    int original_width = gradient.shape[width_dim];

    NodeContext grad = gradient;

    int pad_left = padding[0], pad_right = padding[1];
    int length_width = original_width - pad_left - pad_right;

    // Remove width padding
    graphlib::OpType narrow_width_op("narrow", {width_dim, pad_left, length_width, original_width});
    narrow_width_op.set_attr("dim", width_dim);
    narrow_width_op.set_attr("start", pad_left);
    narrow_width_op.set_attr("length", length_width);
    narrow_width_op.set_attr("original_length", original_width);

    grad = ac.autograd->create_op(ac, narrow_width_op, {grad});

    // Then remove height padding if it exists
    if (padding.size() == 4)
    {
        int pad_top = padding[2];
        int pad_bottom = padding[3];

        int length_height = original_height - pad_top - pad_bottom;

        graphlib::OpType narrow_height_op("narrow", {height_dim, pad_top, length_height, original_height});
        narrow_height_op.set_attr("dim", height_dim);
        narrow_height_op.set_attr("start", pad_top);
        narrow_height_op.set_attr("length", length_height);
        narrow_height_op.set_attr("original_length", original_height);

        grad = ac.autograd->create_op(ac, narrow_height_op, {grad});
    }

    return grad;
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Pad, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Pad should have exactly 1 input");
    TT_ASSERT(op.attrs().size() == 4, "Pad should have 4 attributes: padding, mode, value, channel_last");

    auto padding = op.attr_as<std::vector<int>>("padding");
    auto mode = op.attr_as<int>("mode");
    auto value = op.attr_as<float>("value");
    auto channel_last = op.attr_as<bool>("channel_last");

    // Check if all padding values are 0 - if so, replace with Nop
    bool all_zero = std::all_of(padding.begin(), padding.end(), [](int x) { return x == 0; });
    if (all_zero)
    {
        auto nop_result = dc.op(graphlib::OpType("nop"), {inputs[0]});
        dc.fuse(nop_result);
        return;
    }

    NodeContext activations = inputs[0];
    int shape_size = static_cast<int>(activations.shape.size());

    // Determine dimension axes based on channel_last format
    int height_dim, width_dim;
    if (channel_last)
    {
        height_dim = shape_size - 3;  // height at -3
        width_dim = shape_size - 2;   // width at -2
    }
    else
    {
        height_dim = shape_size - 2;  // height at -2
        width_dim = shape_size - 1;   // width at -1
    }

    // Parse padding values - format is [left, right, top, bottom] or [left, right]
    TT_ASSERT(padding.size() == 2 || padding.size() == 4, "Not supported padding type");

    // int left = padding[0], right = padding[1], top = 0, bottom = 0;
    // if (padding.size() == 4)
    // {
    //     top = padding[2];
    //     bottom = padding[3];
    // }

    if (mode == 0)  // constant mode
    {
        // Use ConstantPad operation with direct TTIR mapping
        NodeContext result = create_constant_pad_op(dc, activations, padding, height_dim, width_dim, value);
        dc.fuse(result);
    }
    else if (mode == 1)  // replicate mode
    {
        decompose_replicate_mode(dc, activations, padding, height_dim, width_dim);
    }
    else
    {
        // For reflect mode (mode=2), delegate to Python for now
        // Reflect mode requires complex tensor creation for index mirroring
        // which is difficult to implement correctly in C++ decomposition context
        return op.base_decompose(old_op_type, "get_f_forge_decompose", dc, inputs);
    }
}

NodeContext create_pad(DecomposingContext &dc, const std::vector<int> &shape, float value, DataFormat data_format)
{
    // Convert int vector to int64_t for PyTorch
    std::vector<int64_t> torch_shape(shape.begin(), shape.end());
    at::Tensor tensor = at::full(torch_shape, value, at::TensorOptions().dtype(at::kFloat));

    // Use the new create_constant_tensor method following PR #2619 pattern
    return DecomposingContext::create_constant_tensor(dc, tensor);
}

// Following Python: def extract(dc, input, dim_axis, start, stop)
NodeContext extract(DecomposingContext &dc, const NodeContext &input, int dim_axis, int start, int stop)
{
    graphlib::OpType index_op("index", {dim_axis, start, stop, 1});
    index_op.set_attr("dim", dim_axis);
    index_op.set_attr("start", start);
    index_op.set_attr("stop", stop);
    index_op.set_attr("stride", 1);

    return dc.op(index_op, {input});
}

// Following Python: def repeat_vector(dc, input, n_repeats, axis)
NodeContext repeat_vector(DecomposingContext &dc, const NodeContext &input, int n_repeats, int axis)
{
    // Create repeats vector - all dimensions have 1 repeat except the axis
    std::vector<int> repeats(input.shape.size(), 1);
    repeats[axis] = n_repeats;

    std::vector<graphlib::OpType::Attr> repeat_attrs(repeats.begin(), repeats.end());

    graphlib::OpType repeat_op("repeat", repeat_attrs, {{"repeats", repeats}});

    return dc.op(repeat_op, {input});
}

// Note: extract_and_mirror removed - reflect mode delegates to Python

NodeContext concat_patches(
    DecomposingContext &dc,
    const NodeContext *first_patch,
    const NodeContext &center,
    const NodeContext *second_patch,
    int dim_axis)
{
    std::vector<NodeContext> inputs;

    if (first_patch)
    {
        inputs.push_back(*first_patch);
    }

    inputs.push_back(center);

    if (second_patch)
    {
        inputs.push_back(*second_patch);
    }

    if (inputs.size() == 1)
    {
        return center;  // No concatenation needed
    }

    graphlib::OpType concat_op("concatenate", {}, {{"dim", dim_axis}});
    return dc.op(concat_op, inputs);
}

NodeContext create_constant_pad_op(
    DecomposingContext &dc,
    const NodeContext &input,
    const std::vector<int> &padding,
    int height_dim,
    int width_dim,
    float value)
{
    // Convert padding format from [left, right] or [left, right, top, bottom]
    // to TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]

    int rank = static_cast<int>(input.shape.size());
    std::vector<int> constant_padding(rank * 2, 0);  // Initialize all to 0

    int left = padding[0], right = padding[1];

    constant_padding[width_dim * 2] = left;       // low padding
    constant_padding[width_dim * 2 + 1] = right;  // high padding

    if (padding.size() == 4)
    {
        int top = padding[2], bottom = padding[3];

        constant_padding[height_dim * 2] = top;         // low padding
        constant_padding[height_dim * 2 + 1] = bottom;  // high padding
    }

    // Create constant_pad operation for direct TTIR mapping
    return dc.op(graphlib::OpType("constant_pad", {}, {{"padding", constant_padding}, {"value", value}}), {input});
}

void decompose_replicate_mode(
    DecomposingContext &dc,
    const NodeContext &activations,
    const std::vector<int> &padding,
    int height_dim,
    int width_dim)
{
    NodeContext result = activations;

    // Parse padding values - format is [left, right] or [left, right, top, bottom]
    int left = padding[0], right = padding[1], top = 0, bottom = 0;
    if (padding.size() == 4)
    {
        top = padding[2];
        bottom = padding[3];
    }

    std::unique_ptr<NodeContext> left_pad, right_pad, top_pad, bot_pad;

    // Apply width padding
    if (left > 0)
    {
        NodeContext left_patch = extract(dc, result, width_dim, 0, 1);
        left_pad = std::make_unique<NodeContext>(repeat_vector(dc, left_patch, left, width_dim));
    }
    if (right > 0)
    {
        int width_size = result.shape[width_dim];
        NodeContext right_patch = extract(dc, result, width_dim, width_size - 1, width_size);
        right_pad = std::make_unique<NodeContext>(repeat_vector(dc, right_patch, right, width_dim));
    }

    result = concat_patches(dc, left_pad.get(), result, right_pad.get(), width_dim);

    // Apply height padding
    if (top > 0)
    {
        NodeContext top_patch = extract(dc, result, height_dim, 0, 1);
        top_pad = std::make_unique<NodeContext>(repeat_vector(dc, top_patch, top, height_dim));
    }
    if (bottom > 0)
    {
        int height_size = result.shape[height_dim];
        NodeContext bot_patch = extract(dc, result, height_dim, height_size - 1, height_size);
        bot_pad = std::make_unique<NodeContext>(repeat_vector(dc, bot_patch, bottom, height_dim));
    }

    result = concat_patches(dc, top_pad.get(), result, bot_pad.get(), height_dim);
    dc.fuse(result);
}

}  // namespace pad
}  // namespace ops
}  // namespace tt
