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
namespace resize_2d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize2d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Resize2d expects 1 input tensor");

    auto sizes = op.attr_as<std::vector<int>>("sizes");
    TT_ASSERT(sizes.size() == 2, "Resize2d sizes must have 2 elements");
    std::string mode = op.attr_as<std::string>("mode");
    bool channel_last = op.attr_as<bool>("channel_last");
    bool align_corners = op.attr_as<bool>("align_corners");

    at::Tensor activation = tensors[0];

    if (channel_last)
        activation = activation.permute({0, 3, 1, 2});

    torch::nn::functional::InterpolateFuncOptions options = torch::nn::functional::InterpolateFuncOptions();
    options = options.size(std::vector<int64_t>{sizes[0], sizes[1]});

    // align_corners option can be only set to bilinear interpolation mode
    if (align_corners && mode != "nearest")
        options = options.align_corners(align_corners);

    if (mode == "nearest")
        options = options.mode(torch::kNearest);
    else if (mode == "bilinear")
        options = options.mode(torch::kBilinear);
    else
        TT_THROW("OpType::Resize2d does not support {} interpolation mode", mode);

    at::Tensor result;
    result = torch::nn::functional::interpolate(activation, options);

    if (channel_last)
        result = result.permute({0, 2, 3, 1});

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize2d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Resize2d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() == 4, "Resize2d input must have 4 dimensions");

    auto sizes = op.attr_as<std::vector<int>>("sizes");
    TT_ASSERT(sizes.size() == 2, "Resize2d sizes must have 2 elements");
    int size_h = sizes[0];
    int size_w = sizes[1];
    bool channel_last = op.attr_as<bool>("channel_last");

    const size_t rank = input_shape.size();
    size_t h_idx = channel_last ? rank - 3 : rank - 2;
    size_t w_idx = channel_last ? rank - 2 : rank - 1;

    int input_h = static_cast<int>(input_shape[h_idx]);
    int input_w = static_cast<int>(input_shape[w_idx]);

    // Determine whether each spatial dimension will be upsampling (increasing size)
    bool is_upsampling_height = (size_h >= input_h);
    bool is_upsampling_width = (size_w >= input_w);

    // Mixed up/downsample not allowed
    if (is_upsampling_height != is_upsampling_width)
    {
        TT_THROW(
            "OpType::Resize2d does not support one spatial dimension as upsample and another spatial dimension as "
            "downsample");
    }

    if (is_upsampling_height && is_upsampling_width)
        TT_ASSERT(
            (size_h % input_h == 0) && (size_w % input_w == 0), "Only support upsample with integer scale factor");
    else
        TT_ASSERT(
            (input_h % size_h == 0) && (input_w % size_w == 0), "Only support downsample with integer scale factor");

    std::vector<uint32_t> output_shape = input_shape;
    output_shape[h_idx] = static_cast<uint32_t>(size_h);
    output_shape[w_idx] = static_cast<uint32_t>(size_w);

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
    TT_DBG_ASSERT(op.type() == OpType::Resize2d, "Wrong op type.");
    TT_THROW("OpType::Resize2d does not have backward.");
    unreachable();
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize2d, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Resize2d expects 1 input");

    auto sizes = op.attr_as<std::vector<int>>("sizes");
    TT_ASSERT(sizes.size() == 2, "Resize2d sizes must have 2 elements");
    int size_h = sizes[0];
    int size_w = sizes[1];
    std::string mode = op.attr_as<std::string>("mode");
    bool channel_last = op.attr_as<bool>("channel_last");
    bool align_corners = op.attr_as<bool>("align_corners");

    NodeContext result = inputs[0];
    Shape input_shape = result.shape;

    const size_t rank = input_shape.size();
    size_t h_idx = channel_last ? rank - 3 : rank - 2;
    size_t w_idx = channel_last ? rank - 2 : rank - 1;

    int input_h = static_cast<int>(input_shape[h_idx]);
    int input_w = static_cast<int>(input_shape[w_idx]);

    // If the resize2d up/down sampling sizes matches with input height and width, there is no need for up/down sampling
    // operation
    if ((size_h == input_h) && (size_w == input_w))
    {
        result = dc.op(graphlib::OpType("nop"), {result});
        dc.fuse(result);
        return;
    }

    // Determine whether each spatial dimension will be upsampling (increasing size)
    bool is_upsampling_height = (size_h >= input_h);
    bool is_upsampling_width = (size_w >= input_w);

    // Mixed up/downsample not allowed
    if (is_upsampling_height != is_upsampling_width)
    {
        TT_THROW(
            "OpType::Resize2d does not support one spatial dimension as upsample and another spatial dimension as "
            "downsample");
    }

    if (!channel_last)
    {
        // Changing the Layout from NCHW to NHWC as ttir.upsample2d supports only the NHWC layout
        result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {result});
        result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {result});
    }

    if (is_upsampling_height && is_upsampling_width)
    {
        if (align_corners && mode != "nearest")
        {
            TT_THROW("align_corners argument not supported in upsample2d op with {} interpolation mode", mode);
        }
        std::vector<int> scale_factor;
        scale_factor.push_back(size_h / input_h);
        scale_factor.push_back(size_w / input_w);
        result = dc.op(
            graphlib::OpType(
                "upsample2d",
                {scale_factor, mode, true},
                {{"scale_factor", scale_factor}, {"mode", mode}, {"channel_last", true}}),
            {result});
    }
    else
    {
        TT_THROW("Downsample2d is not supported yet.");
        unreachable();
        // TODO: Implement downsample2d
        // int scale_factor = channel_last ? static_cast<int>(input_shape[input_shape.size() - 3]) / sizes[0]
        //                                 : static_cast<int>(input_shape[input_shape.size() - 2]) / sizes[0];
        // result = dc.op(
        //     graphlib::OpType(
        //         "downsample2d",
        //         {scale_factor, resize_method, true},
        //         {{"scale_factor", scale_factor}, {"mode", resize_method}, {"channel_last", true}}),
        //     {result});
    }

    if (!channel_last)
    {
        // Changing the Layout back to NCHW from NHWC after operation
        result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {result});
        result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {result});
    }

    dc.fuse(result);
}

}  // namespace resize_2d
}  // namespace ops
}  // namespace tt
