// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
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
namespace softmax_bw
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 3, "Softmax backward should have three operands.");
    TT_ASSERT(op.attrs().size() == 1, "Softmax backward should have one attribute.");

    const at::Tensor &input = tensors[0];   // the input of the softmax
    const at::Tensor &output = tensors[1];  // the output of the softmax function
    const at::Tensor &grad = tensors[2];    // gradient from the previous layer
    int dim = op.attr_as<int>("dim");       // the dimension by which we do softmax

    TT_ASSERT(input.dim() > dim && dim >= -output.dim(), "Given dimension is out of the shape.");

    // Use pytorch's autograd to compute the backward pass. We should fix this with ops (look at decompose.
    at::Tensor input_clone = input.clone().detach();
    input_clone.requires_grad_(true);
    at::Tensor output_computed = torch::softmax(input_clone, dim);
    output_computed.backward(grad);

    return input_clone.grad();
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::SoftmaxBw, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 3, "Softmax backward should have three operands.");
    TT_ASSERT(op.attrs().size() == 1, "Softmax backward should have one attribute.");

    return {Shape::create(in_shapes[0]), {}};
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
    TT_ASSERT(false, "SoftmaxBw does not have backward.");
    unreachable();
}

void decompose_post_autograd(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_ASSERT(inputs.size() == 3, "Softmax backward should have three operands.");
    TT_ASSERT(op.attrs().size() == 1, "Softmax backward should have one attribute.");

    auto &output = inputs[1];          // the output of the softmax function
    auto &grad = inputs[2];            // gradient from the previous layer
    int dim = op.attr_as<int>("dim");  // the dimension by which we do softmax

    uint32_t idx = dim >= 0 ? dim : -dim - 1;
    TT_ASSERT(idx < output.shape.size(), "Given dimension is out of the shape");

    // Decompose: result = (grad - torch.sum(grad * output, dim=dim, keepdim=True)) * output
    auto grad_out = dc.op(graphlib::OpType("multiply"), {grad, output});
    auto gout_sum =
        dc.op(graphlib::OpType("reduce_sum", {{"dim_arg", std::vector<int>{dim}}, {"keep_dim", true}}), {grad_out});
    auto gout_sub = dc.op(graphlib::OpType("subtract"), {grad, gout_sum});
    auto result = dc.op(graphlib::OpType("multiply"), {gout_sub, output});
    dc.fuse(result);
}

}  // namespace softmax_bw
}  // namespace ops
}  // namespace tt
