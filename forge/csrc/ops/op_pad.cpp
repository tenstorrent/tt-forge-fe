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

// Forward declarations for helper functions - matching Python implementation structure
NodeContext decompose_constant_mode(
    DecomposingContext &dc,
    const NodeContext &input,
    float value,
    int left,
    int right,
    int top,
    int bottom,
    int c_dim_axis,
    int r_dim_axis);

NodeContext create_pad(DecomposingContext &dc, const std::vector<int> &shape, float value, DataFormat data_format);

NodeContext extract(DecomposingContext &dc, const NodeContext &input, int dim_axis, int start, int stop);

NodeContext repeat_vector(DecomposingContext &dc, const NodeContext &input, int n_repeats, int axis);

NodeContext concat_patches(
    DecomposingContext &dc,
    const NodeContext *first_patch,
    const NodeContext &center,
    const NodeContext *second_patch,
    int dim_axis);

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

    if (padding.size() == 4)
    {
        int pad_top = padding[2];
        int pad_bottom = padding[3];

        // Remove height padding first
        graphlib::OpType narrow_height_op(
            "narrow",
            {},
            {{"dim", height_dim},
             {"start", pad_top},
             {"length", original_height - pad_top - pad_bottom},
             {"original_length", original_height}});
        grad = ac.autograd->create_op(ac, narrow_height_op, {grad});

        // Then remove width padding
        graphlib::OpType narrow_width_op(
            "narrow",
            {},
            {{"dim", width_dim},
             {"start", pad_left},
             {"length", original_width - pad_left - pad_right},
             {"original_length", original_width}});
        grad = ac.autograd->create_op(ac, narrow_width_op, {grad});
    }
    else
    {
        // Remove only width padding
        graphlib::OpType narrow_width_op(
            "narrow",
            {},
            {{"dim", width_dim},
             {"start", pad_left},
             {"length", original_width - pad_left - pad_right},
             {"original_length", original_width}});
        grad = ac.autograd->create_op(ac, narrow_width_op, {grad});
    }

    return grad;
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Pad, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Pad should have exactly 1 input");
    TT_ASSERT(op.attrs().size() == 4, "Pad should have 4 attributes: padding, mode, value, channel_last");

    // Extract attributes
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
    int r_dim_axis, c_dim_axis;
    if (channel_last)
    {
        r_dim_axis = shape_size - 3;  // height at -3
        c_dim_axis = shape_size - 2;  // width at -2
    }
    else
    {
        r_dim_axis = shape_size - 2;  // height at -2
        c_dim_axis = shape_size - 1;  // width at -1
    }

    // Parse padding values - format is [left, right, top, bottom] or [left, right]
    TT_ASSERT(padding.size() == 2 || padding.size() == 4, "Not supported padding type");
    int left = padding[0], right = padding[1], top = 0, bottom = 0;
    if (padding.size() == 4)
    {
        top = padding[2];
        bottom = padding[3];
    }

    if (mode == 0)
    {  // constant mode
        NodeContext result =
            decompose_constant_mode(dc, activations, value, left, right, top, bottom, c_dim_axis, r_dim_axis);
        dc.fuse(result);
    }
    else if (mode == 1)
    {  // replicate mode
        NodeContext result = activations;

        std::unique_ptr<NodeContext> left_pad, right_pad, top_pad, bot_pad;

        if (left > 0)
        {
            NodeContext left_patch = extract(dc, result, c_dim_axis, 0, 1);
            left_pad = std::make_unique<NodeContext>(repeat_vector(dc, left_patch, left, c_dim_axis));
        }
        if (right > 0)
        {
            int c = result.shape[c_dim_axis];
            NodeContext right_patch = extract(dc, result, c_dim_axis, c - 1, c);
            right_pad = std::make_unique<NodeContext>(repeat_vector(dc, right_patch, right, c_dim_axis));
        }

        result = concat_patches(dc, left_pad.get(), result, right_pad.get(), c_dim_axis);

        if (top > 0)
        {
            NodeContext top_patch = extract(dc, result, r_dim_axis, 0, 1);
            top_pad = std::make_unique<NodeContext>(repeat_vector(dc, top_patch, top, r_dim_axis));
        }
        if (bottom > 0)
        {
            int r = activations.shape[r_dim_axis];  // Use original height, not updated
            NodeContext bot_patch = extract(dc, result, r_dim_axis, r - 1, r);
            bot_pad = std::make_unique<NodeContext>(repeat_vector(dc, bot_patch, bottom, r_dim_axis));
        }

        result = concat_patches(dc, top_pad.get(), result, bot_pad.get(), r_dim_axis);
        dc.fuse(result);
    }
    else
    {
        // For reflect mode (mode=2), delegate to Python for now
        // Reflect mode requires complex tensor creation for index mirroring
        // which is difficult to implement correctly in C++ decomposition context
        return op.base_decompose(old_op_type, "get_f_forge_decompose", dc, inputs);
    }
}

NodeContext decompose_constant_mode(
    DecomposingContext &dc,
    const NodeContext &input,
    float value,
    int left,
    int right,
    int top,
    int bottom,
    int c_dim_axis,
    int r_dim_axis)
{
    NodeContext result = input;
    DataFormat data_format = input.output_df;

    // Handle width padding (left/right)
    std::vector<int> width_shape(result.shape.as_vector<int>());
    std::unique_ptr<NodeContext> left_pad, right_pad;

    if (left > 0)
    {
        width_shape[c_dim_axis] = left;
        left_pad = std::make_unique<NodeContext>(create_pad(dc, width_shape, value, data_format));
    }
    if (right > 0)
    {
        width_shape[c_dim_axis] = right;
        right_pad = std::make_unique<NodeContext>(create_pad(dc, width_shape, value, data_format));
    }

    result = concat_patches(dc, left_pad.get(), result, right_pad.get(), c_dim_axis);

    // Handle height padding (top/bottom) - use updated result shape
    std::vector<int> height_shape(result.shape.as_vector<int>());
    std::unique_ptr<NodeContext> top_pad, bot_pad;

    if (top > 0)
    {
        height_shape[r_dim_axis] = top;
        top_pad = std::make_unique<NodeContext>(create_pad(dc, height_shape, value, data_format));
    }
    if (bottom > 0)
    {
        height_shape[r_dim_axis] = bottom;
        bot_pad = std::make_unique<NodeContext>(create_pad(dc, height_shape, value, data_format));
    }

    result = concat_patches(dc, top_pad.get(), result, bot_pad.get(), r_dim_axis);

    return result;
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
    graphlib::OpType index_op("index", {}, {{"dim", dim_axis}, {"start", start}, {"stop", stop}, {"stride", 1}});

    return dc.op(index_op, {input});
}

// Following Python: def repeat_vector(dc, input, n_repeats, axis)
NodeContext repeat_vector(DecomposingContext &dc, const NodeContext &input, int n_repeats, int axis)
{
    // Create repeats vector - all dimensions have 1 repeat except the axis
    std::vector<int> repeats(input.shape.size(), 1);
    repeats[axis] = n_repeats;

    graphlib::OpType repeat_op("repeat", {}, {{"repeats", repeats}});

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

}  // namespace pad
}  // namespace ops
}  // namespace tt
