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
namespace select
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Select, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Select should have one operand.");
    TT_ASSERT(op.has_attr("dim"), "Select should have dim attribute.");
    TT_ASSERT(op.has_attr("begin"), "Select should have begin attribute.");
    TT_ASSERT(op.has_attr("length"), "Select should have length attribute.");
    TT_ASSERT(op.has_attr("stride"), "Select should have stride attribute.");

    const auto &input_tensor = tensors[0];
    int dim = op.attr_as<int>("dim");
    int begin = op.attr_as<int>("begin");
    int length = op.attr_as<int>("length");
    int stride = op.attr_as<int>("stride");

    if (dim < 0)
        dim += static_cast<int>(input_tensor.dim());

    TT_ASSERT(dim >= 0 && dim < static_cast<int>(input_tensor.dim()), "Dimension out of range.");

    std::vector<at::Tensor> result_slices;

    std::vector<int64_t> zero_shape(input_tensor.sizes().begin(), input_tensor.sizes().end());
    zero_shape[dim] = 1;
    at::Tensor zero_slice_template = torch::zeros(zero_shape, input_tensor.options()).squeeze(dim);

    for (int offset = 0; offset < input_tensor.size(dim) - begin; offset += stride)
    {
        for (int i = begin; i < begin + length; ++i)
        {
            int index = offset + i;
            if (index < input_tensor.size(dim) || stride == input_tensor.size(dim))
                result_slices.push_back(input_tensor.select(dim, index));
            else
                result_slices.push_back(zero_slice_template.clone());
        }
    }

    if (result_slices.empty())
    {
        std::vector<int64_t> output_shape(input_tensor.sizes().begin(), input_tensor.sizes().end());
        output_shape[dim] = 0;
        return torch::empty(output_shape, input_tensor.options());
    }

    return torch::stack(result_slices, dim);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Select, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Select should have one operand.");
    TT_ASSERT(op.has_attr("dim"), "Select should have dim attribute.");
    TT_ASSERT(op.has_attr("begin"), "Select should have begin attribute.");
    TT_ASSERT(op.has_attr("length"), "Select should have length attribute.");
    TT_ASSERT(op.has_attr("stride"), "Select should have stride attribute.");

    int dim = op.attr_as<int>("dim");
    int begin = op.attr_as<int>("begin");
    int length = op.attr_as<int>("length");
    int stride = op.attr_as<int>("stride");

    const auto &input_shape = in_shapes[0];

    if (dim < 0)
        dim += static_cast<int>(input_shape.size());

    TT_ASSERT(dim >= 0 && dim < static_cast<int>(input_shape.size()), "Dimension out of range.");

    std::vector<std::uint32_t> output_shape = input_shape;

    int input_dim_size = static_cast<int>(input_shape[dim]);
    int remaining_size = input_dim_size - begin;
    int round_up_div_result = (remaining_size + stride - 1) / stride;  // equivalent to round_up_div
    output_shape[dim] = static_cast<std::uint32_t>(length * std::max(0, round_up_div_result));

    return {Shape::create(output_shape), {}};
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
    TT_DBG_ASSERT(op.type() == OpType::Select, "Wrong op type.");
    TT_ASSERT(operand == 0, "Invalid operand index for select backward");
    TT_ASSERT(inputs.size() == 1, "Select should have one input");
    TT_ASSERT(op.has_attr("dim"), "Select should have dim attribute.");
    TT_ASSERT(op.has_attr("begin"), "Select should have begin attribute.");
    TT_ASSERT(op.has_attr("length"), "Select should have length attribute.");
    TT_ASSERT(op.has_attr("stride"), "Select should have stride attribute.");

    int dim = op.attr_as<int>("dim");
    int begin = op.attr_as<int>("begin");
    int length = op.attr_as<int>("length");
    int stride = op.attr_as<int>("stride");

    const auto &input_shape = inputs[0].shape;
    std::vector<int> shape_vec = input_shape.as_vector<int>();

    if (dim < 0)
        dim += static_cast<int>(shape_vec.size());

    int orig_size = shape_vec[dim];
    int current_size = gradient.shape.as_vector<int>()[dim];

    NodeContext grad_return = gradient;
    bool grad_return_initialized = false;
    int grad_offset = 0;

    for (int offset = 0; offset < orig_size; offset += stride)
    {
        if (begin > 0)
        {
            std::vector<int64_t> zero_pre_pad_shape;
            for (int val : shape_vec)
            {
                zero_pre_pad_shape.push_back(static_cast<int64_t>(val));
            }
            zero_pre_pad_shape[dim] = static_cast<int64_t>(std::min(begin, orig_size - offset));

            NodeContext zero_slice = ac.autograd->create_constant_tensor(ac, torch::zeros(zero_pre_pad_shape));

            if (!grad_return_initialized)
            {
                grad_return = zero_slice;
                grad_return_initialized = true;
            }
            else
            {
                grad_return = ac.autograd->create_op(
                    ac, graphlib::OpType("concatenate", {}, {{"dim", dim}}), {grad_return, zero_slice});
            }
        }

        if (offset + begin >= orig_size)
            break;

        // Pass the gradient for selected part
        NodeContext grad_slice = ac.autograd->create_op(
            ac,
            graphlib::OpType(
                "select", {}, {{"dim", dim}, {"begin", grad_offset}, {"length", length}, {"stride", current_size}}),
            {gradient});

        if (!grad_return_initialized)
        {
            grad_return = grad_slice;
            grad_return_initialized = true;
        }
        else
        {
            grad_return = ac.autograd->create_op(
                ac, graphlib::OpType("concatenate", {}, {{"dim", dim}}), {grad_return, grad_slice});
        }

        grad_offset += length;
        if (offset + begin + length >= orig_size)
            break;

        int zero_padding_length = stride - length - begin;
        if (zero_padding_length > 0)
        {
            std::vector<int64_t> zero_post_pad_shape;
            for (int val : shape_vec) zero_post_pad_shape.push_back(static_cast<int64_t>(val));

            zero_post_pad_shape[dim] =
                static_cast<int64_t>(std::min(zero_padding_length, orig_size - offset - begin - length));

            if (zero_post_pad_shape[dim] > 0)
            {
                NodeContext zero_slice = ac.autograd->create_constant_tensor(ac, torch::zeros(zero_post_pad_shape));

                grad_return = ac.autograd->create_op(
                    ac, graphlib::OpType("concatenate", {}, {{"dim", dim}}), {grad_return, zero_slice});
            }
        }
    }

    if (!grad_return_initialized)
    {
        std::vector<int64_t> zero_shape;
        for (int val : shape_vec) zero_shape.push_back(static_cast<int64_t>(val));

        grad_return = ac.autograd->create_constant_tensor(ac, torch::zeros(zero_shape));
    }

    return grad_return;
}

}  // namespace select
}  // namespace ops
}  // namespace tt
