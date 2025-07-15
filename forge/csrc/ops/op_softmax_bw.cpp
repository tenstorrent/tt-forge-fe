// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

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
namespace softmax_bw
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 3, "Softmax backward should have three operands.");
    TT_ASSERT(op.attrs().size() == 1, "Softmax backward should have one attribute.");

    at::Tensor input = tensors[0];          // the input of the softmax
    const at::Tensor &output = tensors[1];  // the output of the softmax function
    const at::Tensor &grad = tensors[2];    // gradient from the previous layer
    int dim = op.attr_as<int>("dim");       // the dimension by which we do softmax

    TT_ASSERT(input.dim() > dim && dim >= -output.dim(), "Given dimension is out of the shape.");

    // Use PyTorch's autograd to compute the backward pass
    input.requires_grad_(true);
    auto output_computed = torch::softmax(input, dim);
    output_computed.backward(grad);

    return input.grad();
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::SoftmaxBw, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 3, "Softmax backward should have three operands.");
    TT_ASSERT(op.attrs().size() == 1, "Softmax backward should have one attribute.");

    // Return the shape of the first input (the input of the softmax)
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>());
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_ASSERT(false, "SoftmaxBw does not have backward.");
    unreachable();
}

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_ASSERT(inputs.size() == 3, "Softmax backward should have three operands.");
    TT_ASSERT(op.attrs().size() == 1, "Softmax backward should have one attribute.");

    auto &output = inputs[1];          // the output of the softmax function
    auto &grad = inputs[2];            // gradient from the previous layer
    int dim = op.attr_as<int>("dim");  // the dimension by which we do softmax

    TT_ASSERT(
        output.shape.size() > static_cast<size_t>(dim) && dim >= -static_cast<int>(output.shape.size()),
        "Given dimension is out of the shape");

    // Decompose: result = (grad - torch.sum(grad * output, dim=dim, keepdim=True)) * output
    auto grad_out = dc.op(graphlib::OpType("multiply"), {grad, output});
    auto gout_sum = dc.op(
        graphlib::OpType("reduce_sum", {dim, true}, {{"dim_arg", std::vector<int>{dim}}, {"keep_dim", true}}),
        {grad_out});
    auto gout_sub = dc.op(graphlib::OpType("subtract"), {grad, gout_sum});
    auto result = dc.op(graphlib::OpType("multiply"), {gout_sub, output});
    dc.fuse(result);
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_ASSERT(inputs.size() == 3, "Softmax backward should have three inputs");

    auto shape_tuple = softmax_bw::shape(op, inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1L, std::multiplies<long>());
}

}  // namespace softmax_bw
}  // namespace ops
}  // namespace tt
