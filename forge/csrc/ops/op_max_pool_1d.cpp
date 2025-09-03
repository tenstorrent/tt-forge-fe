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
namespace max_pool_1d
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool1d, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "MaxPool1d expects 1 input tensor");

    const at::Tensor &activations = tensors[0];

    int kernel_size = op.attr_as<int>("kernel_size");
    int stride = op.attr_as<int>("stride");
    int dilation = op.attr_as<int>("dilation");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");
    int padding = op.attr_as<int>("padding");

    at::Tensor padded_activations = torch::nn::functional::pad(
        activations, torch::nn::functional::PadFuncOptions({padding, padding}).value(-INFINITY));

    at::Tensor result = torch::nn::functional::max_pool1d(
        padded_activations,
        torch::nn::functional::MaxPool1dFuncOptions(kernel_size)
            .stride(stride)
            .padding(0)
            .dilation(dilation)
            .ceil_mode(ceil_mode));

    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::MaxPool1d, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "MaxPool1d expects 1 input shape");

    const auto &input_shape = in_shapes[0];
    TT_ASSERT(input_shape.size() >= 3, "MaxPool1d input must have at least 3 dimensions");

    int kernel_size = op.attr_as<int>("kernel_size");
    int stride = op.attr_as<int>("stride");
    int dilation = op.attr_as<int>("dilation");
    int padding = op.attr_as<int>("padding");
    bool ceil_mode = op.attr_as<bool>("ceil_mode");

    TT_ASSERT(dilation == 1, "Currently only support dilation = 1");

    uint32_t l_in = input_shape[input_shape.size() - 1];

    uint32_t l_out;
    if (ceil_mode)
    {
        l_out = static_cast<uint32_t>(
            std::ceil(static_cast<float>(l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1));
    }
    else
    {
        l_out = static_cast<uint32_t>(
            std::floor(static_cast<float>(l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1));
    }

    std::vector<uint32_t> output_shape = input_shape;
    output_shape[output_shape.size() - 1] = l_out;

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
    TT_DBG_ASSERT(op.type() == OpType::MaxPool1d, "Wrong op type.");
    TT_THROW("OpType::MaxPool1d does not have backward.");
    unreachable();
}

}  // namespace max_pool_1d
}  // namespace ops
}  // namespace tt
