// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
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
namespace avg_pool_1d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool1d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "AvgPool1d expects 1 input tensor");

    const at::Tensor &activations = tensors[0];

    int kernel_size = op.attr_as<int>("kernel_size");
    int stride = op.attr_as<int>("stride");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    int padding_left = op.attr_as<int>("padding_left");
    int padding_right = op.attr_as<int>("padding_right");
    bool count_include_pad = op.attr_as<bool>("count_include_pad");

    TT_ASSERT(padding_left == padding_right, "AvgPool1d padding must be symmetric");
    int padding = padding_left;

    at::Tensor result = torch::nn::functional::avg_pool1d(
        activations,
        torch::nn::functional::AvgPool1dFuncOptions(kernel_size)
            .stride(stride)
            .padding(padding)
            .ceil_mode(ceil_mode)
            .count_include_pad(count_include_pad));

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool1d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "AvgPool1d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() >= 3, "AvgPool1d input must have at least 3 dimensions");

    int kernel_size = op.attr_as<int>("kernel_size");
    int stride = op.attr_as<int>("stride");
    int dilation = op.attr_as<int>("dilation");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    int padding_left = op.attr_as<int>("padding_left");

    TT_ASSERT(dilation == 1, "Currently only support dilation = 1");

    uint32_t l_in = input_shape[input_shape.size() - 1];
    uint32_t l_out;
    if (ceil_mode)
    {
        l_out = static_cast<uint32_t>(
            std::ceil(static_cast<float>(l_in + 2 * padding_left - dilation * (kernel_size - 1) - 1) / stride + 1));
    }
    else
    {
        l_out = static_cast<uint32_t>(
            std::floor(static_cast<float>(l_in + 2 * padding_left - dilation * (kernel_size - 1) - 1) / stride + 1));
    }

    std::vector<uint32_t> output_shape = input_shape;
    output_shape[output_shape.size() - 1] = l_out;

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
    TT_DBG_ASSERT(op.type() == OpType::AvgPool1d, "Wrong op type.");
    TT_THROW("OpType::AvgPool1d does not have backward.");
    unreachable();
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::AvgPool1d, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "AvgPool1d expects 1 input");

    int kernel_size = op.attr_as<int>("kernel_size");
    NodeContext activations = inputs[0];

    // Check if this is global pooling (kernel size matches input width)
    if (kernel_size != static_cast<int>(activations.shape[activations.shape.size() - 1]))
    {
        TT_THROW("Only support global avg_pool1d for now");
        unreachable();
    }

    NodeContext reduce_avg = dc.op(
        graphlib::OpType("reduce_avg", {}, {{"dim_arg", std::vector<int>{-1}}, {"keep_dim", true}}), {activations});
    dc.fuse(reduce_avg);
    return;
}

}  // namespace avg_pool_1d
}  // namespace ops
}  // namespace tt
