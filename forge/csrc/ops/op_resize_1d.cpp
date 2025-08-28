// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace resize_1d
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize1d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Resize1d expects 1 input tensor");

    auto size = op.attr_as<int>("size");
    std::string mode = op.attr_as<std::string>("mode");
    bool channel_last = op.attr_as<bool>("channel_last");
    bool align_corners = op.attr_as<bool>("align_corners");

    at::Tensor activation = tensors[0];

    if (channel_last)
        activation = activation.permute({0, 2, 1});

    torch::nn::functional::InterpolateFuncOptions options = torch::nn::functional::InterpolateFuncOptions();
    options = options.size(std::vector<int64_t>{size});

    // align_corners option can be only set to linear interpolation mode
    if (align_corners && mode != "nearest")
        options = options.align_corners(align_corners);

    if (mode == "nearest")
        options = options.mode(torch::kNearest);
    else if (mode == "linear")
        options = options.mode(torch::kLinear);
    else
        TT_THROW("OpType::Resize1d does not support {} interpolation mode", mode);

    at::Tensor result;
    result = torch::nn::functional::interpolate(activation, options);

    if (channel_last)
        result = result.permute({0, 2, 1});

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize1d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Resize1d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() == 3, "Resize1d input must have 3 dimensions");

    auto size = op.attr_as<int>("size");
    bool channel_last = op.attr_as<bool>("channel_last");

    const size_t rank = input_shape.size();
    size_t w_idx = channel_last ? rank - 2 : rank - 1;

    int input_w = static_cast<int>(input_shape[w_idx]);

    // Determine whether it is upsample or downsample
    bool is_upsampling = (size >= input_w);

    if (is_upsampling)
        TT_ASSERT((size % input_w == 0), "Only support upsample with integer scale factor");
    else
        TT_ASSERT((input_w % size == 0), "Only support downsample with integer scale factor");

    std::vector<uint32_t> output_shape = input_shape;
    output_shape[w_idx] = static_cast<uint32_t>(size);

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
    TT_DBG_ASSERT(op.type() == OpType::Resize1d, "Wrong op type.");
    TT_THROW("OpType::Resize1d does not have backward.");
    unreachable();
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize1d, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Resize1d expects 1 input");

    auto size = op.attr_as<int>("size");
    std::string mode = op.attr_as<std::string>("mode");
    bool channel_last = op.attr_as<bool>("channel_last");
    bool align_corners = op.attr_as<bool>("align_corners");

    NodeContext result = inputs[0];
    Shape input_shape = result.shape;

    const size_t rank = input_shape.size();
    int w_dim = channel_last ? rank - 2 : rank - 1;

    int input_w = static_cast<int>(input_shape[w_dim]);

    // If the resize1d up/down sampling size matches with input width, there is no need for up/down sampling operation
    if (size == input_w)
    {
        result = dc.op(Op("nop"), {result});
        dc.fuse(result);
        return;
    }

    // Determine whether it is upsample or downsample
    bool is_upsampling = (size >= input_w);
    if (!is_upsampling)
    {
        TT_THROW("Resize1d doesn't support downsampling");
        unreachable();
    }

    // ---------------------------------------------------------------------
    // Decompose resize1d into upsample2d by inserting a singleton spatial dim
    //
    // Goal: convert 1D width-based resize into a 2D operation where the
    //       added spatial dimension (height) == 1. This allows reuse of
    //       the existing 2D upsample primitive.
    //
    // Behavior:
    //  - channel_last == false (N, C, W)
    //      Unsqueeze at w_dim -> (N, C, 1, W)
    //  - channel_last == true  (N, W, C)
    //      Unsqueeze at w_dim -> (N, 1, W, C)
    //
    // After inserting the singleton, convert the tensor layout to NHWC
    // because ttnn.Upsample2d expects NHWC.
    // The sequence of transposes below transforms:
    //   (N, C, 1, W) --> (N, 1, W, C)
    // ---------------------------------------------------------------------
    result = dc.op(Op("unsqueeze", {{"dim", w_dim}}), {result});

    if (!channel_last)
    {
        // Changing the Layout from NCHW to NHWC as ttir.upsample2d supports only the NHWC layout
        result = dc.op(Op("transpose", {{"dim0", -3}, {"dim1", -2}}), {result});
        result = dc.op(Op("transpose", {{"dim0", -2}, {"dim1", -1}}), {result});
    }

    TT_ASSERT(
        (mode == "nearest") || (mode == "linear" && !align_corners),
        "align_corners argument not supported in upsample2d op with {} interpolation mode",
        mode);

    // Create a 2D scale_factor vector where:
    //  - first element corresponds to the added singleton spatial dimension (height) -> 1 (no change)
    //  - second element is the width scaling ratio (new_width / old_width)
    // This maps the 1D scaling into a 2D upsample: [1, width_ratio]
    std::vector<int> scale_factor;
    scale_factor.push_back(1);
    scale_factor.push_back(size / input_w);

    // If the mode was "linear" for 1D, map it to "bilinear" for 2D upsample
    mode = (mode == "linear") ? "bilinear" : mode;

    result =
        dc.op(Op("upsample2d", {{"scale_factor", scale_factor}, {"mode", mode}, {"channel_last", true}}), {result});

    if (!channel_last)
    {
        // Convert layout back from NHWC-like (N, 1, New_W, C) -> NCHW-like (N, C, 1, New_W).
        // Reverse the transposes applied before the upsample2d.
        result = dc.op(Op("transpose", {{"dim0", -2}, {"dim1", -1}}), {result});
        result = dc.op(Op("transpose", {{"dim0", -3}, {"dim1", -2}}), {result});
    }

    // ---------------------------------------------------------------------
    // Remove the singleton spatial dimension that was inserted earlier.
    //
    // Before squeeze:
    //  - channel_last == true  : (N, 1, New_W, C) -> squeeze dim w_dim -> (N, New_W, C)
    //  - channel_last == false : (N, C, 1, New_W) -> squeeze dim w_dim -> (N, C, New_W)
    //
    // Note: we squeeze the same dimension index (w_dim) that we used when
    // we first inserted the singleton. That restores the tensor to the
    // original 3D shape but with the new width.
    // ---------------------------------------------------------------------
    result = dc.op(Op("squeeze", {{"dim", w_dim}}), {result});

    dc.fuse(result);
}

}  // namespace resize_1d
}  // namespace ops
}  // namespace tt
