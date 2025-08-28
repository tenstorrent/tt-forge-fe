// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace constant_pad
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::ConstantPad, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "ConstantPad should have exactly 1 input tensor");
    TT_ASSERT(op.attrs().size() == 2, "ConstantPad should have 2 attributes: padding, value");

    auto padding = op.attr_as<std::vector<int>>("padding");
    auto value = op.attr_as<float>("value");

    at::Tensor input_tensor = tensors[0];

    // PyTorch expects padding in reverse dimension order
    std::vector<int64_t> pytorch_padding;
    for (int i = padding.size() - 2; i >= 0; i -= 2)
    {
        pytorch_padding.push_back(padding[i]);      // low
        pytorch_padding.push_back(padding[i + 1]);  // high
    }

    return at::pad(input_tensor, pytorch_padding, "constant", value);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::ConstantPad, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "ConstantPad should have exactly 1 input shape");
    TT_ASSERT(op.attrs().size() == 2, "ConstantPad should have 2 attributes: padding, value");

    auto padding = op.attr_as<std::vector<int>>("padding");
    auto shape = in_shapes[0];

    TT_ASSERT(padding.size() == shape.size() * 2, "Padding should have rank*2 elements");

    // Apply padding to each dimension
    // padding format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]
    for (size_t i = 0; i < shape.size(); ++i)
    {
        int low_pad = padding[i * 2];
        int high_pad = padding[i * 2 + 1];
        shape[i] += low_pad + high_pad;
    }

    return std::make_tuple(graphlib::Shape::create(shape), std::vector<graphlib::DimBroadcast>{});
}

graphlib::NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<graphlib::NodeContext> &inputs,
    const graphlib::NodeContext &output,
    const graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::ConstantPad, "Wrong op type.");
    TT_ASSERT(operand == 0, "ConstantPad should have exactly 1 operand");
    TT_ASSERT(op.attrs().size() == 2, "ConstantPad should have 2 attributes: padding, value");

    auto padding = op.attr_as<std::vector<int>>("padding");

    TT_ASSERT(padding.size() == gradient.shape.size() * 2, "Padding should have rank*2 elements");

    // Check if all padding values are 0 - if so, return gradient as is
    bool all_zero = std::all_of(padding.begin(), padding.end(), [](int x) { return x == 0; });
    if (all_zero)
    {
        return gradient;
    }

    graphlib::NodeContext grad = gradient;

    int padding_dims = static_cast<int>(padding.size() / 2);

    // Remove padding by applying index operations for each dimension
    for (int dim = 0; dim < padding_dims; dim++)
    {
        int low_pad = padding[dim * 2];
        int high_pad = padding[dim * 2 + 1];

        if (low_pad > 0 || high_pad > 0)
        {
            int original_size = grad.shape[dim];
            int new_size = original_size - low_pad - high_pad;
            int stop = low_pad + new_size;

            Op index_op("index");
            index_op.set_attr("dim", dim);
            index_op.set_attr("start", low_pad);
            index_op.set_attr("stop", stop);
            index_op.set_attr("stride", 1);

            grad = ac.autograd->create_op(ac, index_op, {grad});
        }
    }

    return grad;
}

}  // namespace constant_pad
}  // namespace ops
}  // namespace tt
