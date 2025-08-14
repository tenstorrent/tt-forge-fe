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
namespace upsample_2d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Upsample2d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Upsample2d expects 1 input tensor");

    std::string mode = op.attr_as<std::string>("mode");
    bool channel_last = op.attr_as<bool>("channel_last");

    at::Tensor activation = tensors[0];
    if (channel_last)
        activation = activation.permute({0, 3, 1, 2});

    torch::nn::functional::InterpolateFuncOptions options = torch::nn::functional::InterpolateFuncOptions();

    Attr attr = op.get_attr("scale_factor");

    std::vector<double> scale_factor;

    std::visit(
        [&](auto &&v)
        {
            using U = std::decay_t<decltype(v)>;

            if constexpr (std::is_same_v<U, int>)
            {
                scale_factor = {static_cast<double>(v), static_cast<double>(v)};
            }
            else if constexpr (std::is_same_v<U, std::vector<int>>)
            {
                scale_factor.reserve(v.size());
                for (int x : v) scale_factor.push_back(static_cast<double>(x));
            }
            else
            {
                TT_THROW("Unsupported scale_factor type");
            }
        },
        attr);

    options.scale_factor(scale_factor);

    if (mode == "nearest")
        options = options.mode(torch::kNearest);
    else if (mode == "bilinear")
        options = options.mode(torch::kBilinear);
    else
        TT_THROW("OpType::Upsample2d does not support {} interpolation mode", mode);

    at::Tensor result = torch::nn::functional::interpolate(activation, options);

    if (channel_last)
        result = result.permute({0, 2, 3, 1});

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Upsample2d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Upsample2d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() == 4, "Upsample2d input must have at least 4 dimensions");

    Attr attr = op.get_attr("scale_factor");

    std::vector<uint32_t> scale_factor_vec;

    std::visit(
        [&](auto &&v)
        {
            using U = std::decay_t<decltype(v)>;

            if constexpr (std::is_same_v<U, int> || std::is_same_v<U, unsigned int>)
            {
                scale_factor_vec = {static_cast<uint32_t>(v), static_cast<uint32_t>(v)};
            }
            else if constexpr (std::is_same_v<U, std::vector<int>>)
            {
                scale_factor_vec.reserve(v.size());
                for (int x : v) scale_factor_vec.push_back(static_cast<uint32_t>(x));
            }
            else if constexpr (std::is_same_v<U, std::vector<uint32_t>>)
            {
                scale_factor_vec = v;
            }
            else
            {
                TT_THROW("Unsupported scale_factor type");
            }
        },
        attr);

    bool channel_last = op.attr_as<bool>("channel_last");

    std::vector<uint32_t> output_shape = input_shape;

    if (channel_last)
    {
        // Input: [N, ..., H, W, C], output: [N, ..., scale*H, scale*W, C]
        output_shape[output_shape.size() - 3] *= scale_factor_vec[0];
        output_shape[output_shape.size() - 2] *= scale_factor_vec[1];
    }
    else
    {
        // Input: [N, C, ..., H, W], output: [N, C, ..., scale*H, scale*W]
        output_shape[output_shape.size() - 2] *= scale_factor_vec[0];
        output_shape[output_shape.size() - 1] *= scale_factor_vec[1];
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
    TT_DBG_ASSERT(op.type() == OpType::Upsample2d, "Wrong op type.");
    TT_THROW("OpType::Upsample2d does not have backward.");
    unreachable();
}

}  // namespace upsample_2d
}  // namespace ops
}  // namespace tt
