// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <memory>
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
namespace pad
{
using namespace graphlib;

struct PaddingParams
{
    int left, right, top, bottom;
    int width_dim, height_dim;

    // Parse padding values - format is [left, right] or [left, right, top, bottom]
    // Determine dimension axes based on channel_last format
    PaddingParams(const std::vector<int> &padding, int shape_size, bool channel_last) :
        left(padding[0]), right(padding[1]), top(0), bottom(0)
    {
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

        if (padding.size() == 4)
        {
            top = padding[2];
            bottom = padding[3];
        }
    }
};

NodeContext extract(DecomposingContext &dc, const NodeContext &input, int dim_axis, int start, int stop)
{
    graphlib::OpType index_op("index", {dim_axis, start, stop, 1});
    index_op.set_attr("dim", dim_axis);
    index_op.set_attr("start", start);
    index_op.set_attr("stop", stop);
    index_op.set_attr("stride", 1);

    return dc.op(index_op, {input});
}

NodeContext repeat_vector(DecomposingContext &dc, const NodeContext &input, int n_repeats, int axis)
{
    // Create repeats vector - all dimensions have 1 repeat except the axis
    std::vector<int> repeats(input.shape.size(), 1);
    repeats[axis] = n_repeats;

    std::vector<graphlib::OpType::Attr> repeat_attrs(repeats.begin(), repeats.end());

    graphlib::OpType repeat_op("repeat", repeat_attrs, {{"repeats", repeats}});

    return dc.op(repeat_op, {input});
}

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

NodeContext extract_and_mirror(DecomposingContext &dc, const NodeContext &input, int dim_axis, int start, int stop)
{
    NodeContext patch = extract(dc, input, dim_axis, start, stop);

    // Create indices for mirroring: [stop-start-1, stop-start-2, ..., 1, 0]
    int patch_size = stop - start;
    std::vector<int> indices_vec;
    for (int i = patch_size - 1; i >= 0; i--)
    {
        indices_vec.push_back(i);
    }

    at::Tensor indices_tensor = at::tensor(indices_vec, at::TensorOptions().dtype(at::kLong));
    NodeContext indices = DecomposingContext::create_constant_tensor(dc, indices_tensor);

    // Mirror patch using adv_index
    graphlib::OpType adv_index_op("adv_index", {dim_axis}, {{"dim", dim_axis}});
    NodeContext patch_mirrored = dc.op(adv_index_op, {patch, indices});

    return patch_mirrored;
}

NodeContext create_constant_pad_op(
    DecomposingContext &dc, const NodeContext &input, const PaddingParams &params, float value)
{
    // Convert padding format from [left, right] or [left, right, top, bottom]
    // to TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]

    int rank = static_cast<int>(input.shape.size());
    std::vector<int> constant_padding(rank * 2, 0);  // Initialize all to 0

    constant_padding[params.width_dim * 2] = params.left;       // low padding
    constant_padding[params.width_dim * 2 + 1] = params.right;  // high padding

    if (params.top > 0 || params.bottom > 0)
    {
        constant_padding[params.height_dim * 2] = params.top;         // low padding
        constant_padding[params.height_dim * 2 + 1] = params.bottom;  // high padding
    }

    // Create constant_pad operation for direct TTIR mapping
    return dc.op(graphlib::OpType("constant_pad", {}, {{"padding", constant_padding}, {"value", value}}), {input});
}

void decompose_replicate_mode(DecomposingContext &dc, const NodeContext &input, const PaddingParams &params)
{
    NodeContext result = input;

    std::unique_ptr<NodeContext> left_pad, right_pad, top_pad, bot_pad;

    // Apply width padding
    if (params.left > 0)
    {
        NodeContext left_patch = extract(dc, result, params.width_dim, 0, 1);
        left_pad = std::make_unique<NodeContext>(repeat_vector(dc, left_patch, params.left, params.width_dim));
    }
    if (params.right > 0)
    {
        int width_size = result.shape[params.width_dim];
        NodeContext right_patch = extract(dc, result, params.width_dim, width_size - 1, width_size);
        right_pad = std::make_unique<NodeContext>(repeat_vector(dc, right_patch, params.right, params.width_dim));
    }

    result = concat_patches(dc, left_pad.get(), result, right_pad.get(), params.width_dim);

    // Apply height padding
    if (params.top > 0)
    {
        NodeContext top_patch = extract(dc, result, params.height_dim, 0, 1);
        top_pad = std::make_unique<NodeContext>(repeat_vector(dc, top_patch, params.top, params.height_dim));
    }
    if (params.bottom > 0)
    {
        int height_size = result.shape[params.height_dim];
        NodeContext bot_patch = extract(dc, result, params.height_dim, height_size - 1, height_size);
        bot_pad = std::make_unique<NodeContext>(repeat_vector(dc, bot_patch, params.bottom, params.height_dim));
    }

    result = concat_patches(dc, top_pad.get(), result, bot_pad.get(), params.height_dim);
    dc.fuse(result);
}

void decompose_reflect_mode(DecomposingContext &dc, const NodeContext &input, const PaddingParams &params)
{
    NodeContext result = input;

    // Validate padding limits for reflect mode
    int width_size = result.shape[params.width_dim];
    int height_size = result.shape[params.height_dim];

    TT_ASSERT(
        params.left <= width_size - 1 && params.right <= width_size - 1,
        "Both left padding (" + std::to_string(params.left) + ") and right padding (" + std::to_string(params.right) +
            ") has to be max " + std::to_string(width_size - 1) + " each");

    TT_ASSERT(
        params.top <= height_size - 1 && params.bottom <= height_size - 1,
        "Both top padding (" + std::to_string(params.top) + ") and bottom padding (" + std::to_string(params.bottom) +
            ") has to be max " + std::to_string(height_size - 1) + " each");

    std::unique_ptr<NodeContext> left_pad, right_pad, top_pad, bot_pad;

    // Step 1: Extract left and right patches which are on the width_dim axis and mirror them horizontally
    if (params.left > 0)
    {
        left_pad = std::make_unique<NodeContext>(extract_and_mirror(dc, result, params.width_dim, 1, params.left + 1));
    }
    if (params.right > 0)
    {
        width_size = result.shape[params.width_dim];
        right_pad = std::make_unique<NodeContext>(
            extract_and_mirror(dc, result, params.width_dim, width_size - params.right - 1, width_size - 1));
    }

    // Step 2: Concatenate the mirrored patches to the original result
    result = concat_patches(dc, left_pad.get(), result, right_pad.get(), params.width_dim);

    // Step 3: Extract top and bottom patches which are on the height_dim axis and mirror them vertically
    if (params.top > 0)
    {
        top_pad = std::make_unique<NodeContext>(extract_and_mirror(dc, result, params.height_dim, 1, params.top + 1));
    }
    if (params.bottom > 0)
    {
        height_size = result.shape[params.height_dim];
        bot_pad = std::make_unique<NodeContext>(
            extract_and_mirror(dc, result, params.height_dim, height_size - params.bottom - 1, height_size - 1));
    }

    // Step 4: Concatenate the mirrored patches to the original result
    result = concat_patches(dc, top_pad.get(), result, bot_pad.get(), params.height_dim);
    dc.fuse(result);
}

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
        if (mode == 0)  // constant mode - value parameter is used
        {
            return at::pad(
                input_tensor, std::vector<int64_t>(padding.begin(), padding.end()), mode_options[mode], value);
        }
        // replicate and reflect modes - no value parameter
        return at::pad(input_tensor, std::vector<int64_t>(padding.begin(), padding.end()), mode_options[mode]);
    }

    // When channel_last=True, the input tensor is in format (N, D1, D2, ..., Dn, C)
    // We need to move the channel from the last position to position 1
    // to get (N, C, D1, D2, ..., Dn) which is what PyTorch expects for padding
    int ndim = input_tensor.dim();

    // Create permutation that moves channel from the last dim to position 1
    // For a tensor (N, D1, D2, ..., Dn, C) -> (N, C, D1, D2, ..., Dn)
    std::vector<int64_t> perm = {0, ndim - 1};                 // batch, channel
    for (int64_t i = 1; i < ndim - 1; i++) perm.push_back(i);  // spatial dimensions: 1, 2, ..., ndim-2

    at::Tensor transposed = input_tensor.permute(perm);

    at::Tensor padded;
    if (mode == 0)  // constant mode - value parameter is used
    {
        padded = at::pad(transposed, std::vector<int64_t>(padding.begin(), padding.end()), mode_options[mode], value);
    }
    else  // replicate and reflect modes - no value parameter
    {
        padded = at::pad(transposed, std::vector<int64_t>(padding.begin(), padding.end()), mode_options[mode]);
    }

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
    int shape_size = static_cast<int>(shape.size());

    PaddingParams params(padding, shape_size, channel_last);

    // Apply width padding
    shape[params.width_dim] += params.left + params.right;

    // Apply height padding (if present)
    if (params.top > 0 || params.bottom > 0)
        shape[params.height_dim] += params.top + params.bottom;

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
    PaddingParams params(padding, shape_size, channel_last);

    int original_height = gradient.shape[params.height_dim];
    int original_width = gradient.shape[params.width_dim];

    NodeContext grad = gradient;
    int length_width = original_width - params.left - params.right;

    // Remove width padding
    graphlib::OpType narrow_width_op("narrow", {params.width_dim, params.left, length_width, original_width});
    narrow_width_op.set_attr("dim", params.width_dim);
    narrow_width_op.set_attr("start", params.left);
    narrow_width_op.set_attr("length", length_width);
    narrow_width_op.set_attr("original_length", original_width);

    grad = ac.autograd->create_op(ac, narrow_width_op, {grad});

    // Then remove height padding if it exists
    if (params.top > 0 || params.bottom > 0)
    {
        int length_height = original_height - params.top - params.bottom;

        graphlib::OpType narrow_height_op("narrow", {params.height_dim, params.top, length_height, original_height});
        narrow_height_op.set_attr("dim", params.height_dim);
        narrow_height_op.set_attr("start", params.top);
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

    NodeContext input = inputs[0];
    int shape_size = static_cast<int>(input.shape.size());

    TT_ASSERT(padding.size() == 2 || padding.size() == 4, "Not supported padding type");
    PaddingParams params(padding, shape_size, channel_last);

    if (mode == 0)  // constant mode
    {
        // Use ConstantPad operation with direct TTIR mapping
        NodeContext result = create_constant_pad_op(dc, input, params, value);
        dc.fuse(result);
    }
    else if (mode == 1)  // replicate mode
    {
        decompose_replicate_mode(dc, input, params);
    }
    else  // reflect mode
    {
        decompose_reflect_mode(dc, input, params);
    }
}

}  // namespace pad
}  // namespace ops
}  // namespace tt
