// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>
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
    TT_DBG_ASSERT(op.type() == OpType::Pad, "Wrong op type.");
    TT_ASSERT(operand == 0, "Pad should have exactly 1 operand");
    TT_ASSERT(op.attrs().size() == 4, "Pad should have 4 attributes: padding, mode, value, channel_last");

    // Extract attributes using the correct pattern
    auto padding = op.attr_as<std::vector<int>>("padding");
    auto channel_last = op.attr_as<bool>("channel_last");

    // Backward pass for pad is to remove the padding (narrow operation)
    TT_ASSERT(padding.size() == 2 || padding.size() == 4, "Not supported padding type");

    int height_dim, width_dim;
    if (channel_last)
    {
        height_dim = -3;  // channel_last: height is at -3
        width_dim = -2;   // channel_last: width is at -2
    }
    else
    {
        height_dim = -2;  // channel_first: height is at -2
        width_dim = -1;   // channel_first: width is at -1
    }

    int original_height = gradient.shape[height_dim];
    int original_width = gradient.shape[width_dim];

    NodeContext grad = gradient;

    if (padding.size() == 4)
    {
        int pad_left = padding[0];
        int pad_right = padding[1];
        int pad_top = padding[2];
        int pad_bottom = padding[3];

        // Remove height padding first using autograd create_op
        graphlib::OpType narrow_height_op(
            "narrow", {height_dim, pad_top, original_height - pad_top - pad_bottom, original_height});
        narrow_height_op.set_attr("dim", height_dim);
        narrow_height_op.set_attr("start", pad_top);
        narrow_height_op.set_attr("length", original_height - pad_top - pad_bottom);
        narrow_height_op.set_attr("original_length", original_height);
        grad = ac.autograd->create_op(ac, narrow_height_op, {grad});

        // Then remove width padding
        graphlib::OpType narrow_width_op(
            "narrow", {width_dim, pad_left, original_width - pad_left - pad_right, original_width});
        narrow_width_op.set_attr("dim", width_dim);
        narrow_width_op.set_attr("start", pad_left);
        narrow_width_op.set_attr("length", original_width - pad_left - pad_right);
        narrow_width_op.set_attr("original_length", original_width);
        grad = ac.autograd->create_op(ac, narrow_width_op, {grad});
    }
    else
    {
        int pad_left = padding[0];
        int pad_right = padding[1];

        // Remove width padding only
        graphlib::OpType narrow_width_op(
            "narrow", {width_dim, pad_left, original_width - pad_left - pad_right, original_width});
        narrow_width_op.set_attr("dim", width_dim);
        narrow_width_op.set_attr("start", pad_left);
        narrow_width_op.set_attr("length", original_width - pad_left - pad_right);
        narrow_width_op.set_attr("original_length", original_width);
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
    // auto padding = op.attr_as<std::vector<int>>("padding");
    // auto mode = op.attr_as<int>("mode");
    // auto value = op.attr_as<float>("value");
    // auto channel_last = op.attr_as<bool>("channel_last");

    // // Check if all padding values are 0 - if so, replace with Nop
    // bool all_zero = std::all_of(padding.begin(), padding.end(), [](int x) { return x == 0; });
    // if (all_zero) {
    //     auto nop_result = dc.op("nop", {inputs[0]});
    //     dc.fuse(nop_result);
    //     return;
    // }

    // For now, delegate complex decomposition logic to Python implementation
    // This handles the complex constant/replicate/reflect mode decompositions
    return op.base_decompose(old_op_type, "get_f_forge_decompose", dc, inputs);
}

}  // namespace pad
}  // namespace ops
}  // namespace tt
