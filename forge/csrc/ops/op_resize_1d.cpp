// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
namespace resize_1d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize1d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Resize1d expects 1 input tensor");

    at::Tensor activations = tensors[0];

    int size = op.attr_as<int>("size");
    bool align_corners = op.attr_as<bool>("align_corners");
    bool channel_last = op.attr_as<bool>("channel_last");

    if (channel_last)
    {
        activations = activations.permute({0, 2, 1});
    }

    at::Tensor result = torch::nn::functional::interpolate(
        activations,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{size})
            .mode(torch::kLinear)
            .align_corners(align_corners));

    if (channel_last)
    {
        result = result.permute({0, 2, 1});
    }

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Resize1d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Resize1d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() >= 3, "Resize1d input must have at least 3 dimensions");

    int size = op.attr_as<int>("size");
    bool channel_last = op.attr_as<bool>("channel_last");

    std::vector<uint32_t> output_shape = input_shape;

    if (channel_last)
    {
        // Input: [N, ..., W, C], output: [N, ..., new_W, C]
        output_shape[output_shape.size() - 2] = static_cast<uint32_t>(size);
    }
    else
    {
        // Input: [N, C, ..., W], output: [N, C, ..., new_W]
        output_shape[output_shape.size() - 1] = static_cast<uint32_t>(size);
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
    TT_DBG_ASSERT(op.type() == OpType::Resize1d, "Wrong op type.");
    TT_THROW("OpType::Resize1d does not have backward.");
    unreachable();
}

}  // namespace resize_1d
}  // namespace ops
}  // namespace tt
