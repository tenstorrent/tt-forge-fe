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

    at::Tensor activations = tensors[0];

    auto sizes = op.attr_as<std::vector<int>>("sizes");
    TT_ASSERT(sizes.size() == 2, "Resize2d sizes must have 2 elements");
    int method = op.attr_as<int>("method");
    bool channel_last = op.attr_as<bool>("channel_last");

    std::string resize_method = op_common::get_resize_method(method);

    auto shape = activations.sizes();

    // Determine whether to use upsample or interpolate (matching Python logic)
    bool upsample = channel_last ? sizes[0] >= shape[shape.size() - 3] : sizes[0] >= shape[shape.size() - 2];
    int scale_factor = channel_last ? sizes[0] / shape[shape.size() - 3] : sizes[0] / shape[shape.size() - 2];

    std::cout << "scale_factor " << scale_factor << std::endl;

    if (channel_last)
    {
        activations = activations.permute({0, 3, 1, 2});
    }

    at::Tensor result;

    if (upsample)
    {
        torch::nn::functional::InterpolateFuncOptions options = torch::nn::functional::InterpolateFuncOptions();
        options = options.scale_factor(
            std::vector<double>{static_cast<double>(scale_factor), static_cast<double>(scale_factor)});

        if (resize_method == "nearest")
        {
            options = options.mode(torch::kNearest);
        }
        else if (resize_method == "bilinear")
        {
            options = options.mode(torch::kBilinear);
        }
        else if (resize_method == "cubic")
        {
            options = options.mode(torch::kBicubic);
        }

        result = torch::nn::functional::interpolate(activations, options);
    }
    else
    {
        std::string interp_method = (resize_method == "cubic") ? "bicubic" : resize_method;

        torch::nn::functional::InterpolateFuncOptions options = torch::nn::functional::InterpolateFuncOptions();
        options = options.size(std::vector<int64_t>{sizes[0], sizes[1]});

        if (interp_method == "nearest")
        {
            options = options.mode(torch::kNearest);
        }
        else if (interp_method == "bilinear")
        {
            options = options.mode(torch::kBilinear);
        }
        else if (interp_method == "bicubic")
        {
            options = options.mode(torch::kBicubic);
        }

        result = torch::nn::functional::interpolate(activations, options);
    }

    if (channel_last)
    {
        result = result.permute({0, 2, 3, 1});
    }

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize2d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Resize2d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() >= 4, "Resize2d input must have at least 4 dimensions");

    auto sizes = op.attr_as<std::vector<int>>("sizes");
    TT_ASSERT(sizes.size() == 2, "Resize2d sizes must have 2 elements");
    int size_h = sizes[0];
    int size_w = sizes[1];
    bool channel_last = op.attr_as<bool>("channel_last");

    std::vector<uint32_t> output_shape = input_shape;

    if (channel_last)
    {
        // Input: [N, ..., H, W, C], output: [N, ..., new_H, new_W, C]
        output_shape[output_shape.size() - 3] = static_cast<uint32_t>(size_h);
        output_shape[output_shape.size() - 2] = static_cast<uint32_t>(size_w);
    }
    else
    {
        // Input: [N, C, ..., H, W], output: [N, C, ..., new_H, new_W]
        output_shape[output_shape.size() - 2] = static_cast<uint32_t>(size_h);
        output_shape[output_shape.size() - 1] = static_cast<uint32_t>(size_w);
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
    int method = op.attr_as<int>("method");
    bool channel_last = op.attr_as<bool>("channel_last");

    std::string resize_method = op_common::get_resize_method(method);

    NodeContext result = inputs[0];
    Shape input_shape = result.shape;

    // Determine whether to use upsample or downsample
    bool upsample = channel_last ? sizes[0] >= static_cast<int>(input_shape[input_shape.size() - 3])
                                 : sizes[0] >= static_cast<int>(input_shape[input_shape.size() - 2]);

    if (!channel_last)
    {
        // Changing the Layout from NCHW to NHWC as ttir.upsample2d supports only the NHWC layout
        result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {result});
        result = dc.op(graphlib::OpType("transpose", {}, {{"dim0", -2}, {"dim1", -1}}), {result});
    }

    if (upsample)
    {
        int scale_factor = channel_last ? sizes[0] / static_cast<int>(input_shape[input_shape.size() - 3])
                                        : sizes[0] / static_cast<int>(input_shape[input_shape.size() - 2]);
        result = dc.op(
            graphlib::OpType(
                "upsample2d",
                {scale_factor, resize_method, true},
                {{"scale_factor", scale_factor}, {"mode", resize_method}, {"channel_last", true}}),
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
