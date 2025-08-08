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
namespace layernorm_bw
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::LayernormBw, "Wrong op type.");
    TT_ASSERT(tensors.size() == 5, "Layernorm_bw should have five operands.");

    at::Tensor input = tensors[0];
    at::Tensor gamma = tensors[1];
    at::Tensor beta = tensors[2];
    at::Tensor grad = tensors[3];
    // tensors[4] is the output from forward pass, not needed for PyTorch autograd

    int dim = op.attr_as<int>("dim");
    float epsilon = op.attr_as<float>("epsilon");
    int operand = op.attr_as<int>("operand");

    TT_ASSERT(
        dim == -1 || dim == static_cast<int>(input.dim()) - 1,
        "Normalization can be done only over the last dimension.");
    TT_ASSERT(gamma.size(-1) == input.size(-1), "Weights shape must be the same as normalized shape.");
    TT_ASSERT(beta.size(-1) == input.size(-1), "Bias shape must be the same as normalized shape.");
    TT_ASSERT(operand >= 0 && operand < 3, "Operand index out of range.");

    // Clone tensors and enable gradients
    at::Tensor input_clone = input.clone().detach().requires_grad_(true);
    at::Tensor gamma_clone = gamma.clone().detach().requires_grad_(true);
    at::Tensor beta_clone = beta.clone().detach().requires_grad_(true);

    // Forward pass
    std::vector<int64_t> normalized_shape = {gamma_clone.size(-1)};
    at::Tensor gamma_reshaped = gamma_clone.reshape({gamma_clone.size(-1)});
    at::Tensor beta_reshaped = beta_clone.reshape({beta_clone.size(-1)});

    at::Tensor output = torch::nn::functional::layer_norm(
        input_clone,
        torch::nn::functional::LayerNormFuncOptions(normalized_shape)
            .weight(gamma_reshaped)
            .bias(beta_reshaped)
            .eps(epsilon));

    // Backward pass
    output.backward(grad);

    // Return the appropriate gradient
    if (operand == 0)
    {
        return input_clone.grad();
    }
    else if (operand == 1)
    {
        return gamma_clone.grad().reshape(gamma.sizes());
    }
    else
    {  // operand == 2
        return beta_clone.grad().reshape(beta.sizes());
    }
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::LayernormBw, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 5, "Layernorm_bw should have five operands.");

    int operand = op.attr_as<int>("operand");
    TT_ASSERT(operand >= 0 && operand < 3, "Operand index out of range");

    // Return the shape of the operand we're computing gradients for
    return std::make_tuple(Shape::create(in_shapes[operand]), std::vector<DimBroadcast>{});
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
    TT_DBG_ASSERT(op.type() == OpType::LayernormBw, "Wrong op type.");
    TT_THROW("Layernorm_bw backward should not be called");
    unreachable();
}

void decompose_post_autograd(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::LayernormBw, "Wrong op type.");
    TT_ASSERT(inputs.size() == 5, "Layernorm_bw should have five operands.");

    NodeContext input = inputs[0];
    NodeContext gamma = inputs[1];
    NodeContext beta = inputs[2];
    NodeContext grad = inputs[3];
    NodeContext output = inputs[4];

    int dim = op.attr_as<int>("dim");
    int operand = op.attr_as<int>("operand");

    // Normalize negative dimension
    if (dim < 0)
    {
        dim += static_cast<int>(input.shape.size());
    }

    std::vector<uint32_t> input_shape = input.shape.as_vector<uint32_t>();
    std::vector<uint32_t> gamma_shape = gamma.shape.as_vector<uint32_t>();
    std::vector<uint32_t> beta_shape = beta.shape.as_vector<uint32_t>();

    TT_ASSERT(
        dim == -1 || dim == static_cast<int>(input_shape.size()) - 1,
        "Normalization can be done only over the last dimension.");
    TT_ASSERT(gamma_shape.back() == input_shape.back(), "Weights shape must be the same as normalized shape.");
    TT_ASSERT(beta_shape.back() == input_shape.back(), "Bias shape must be the same as normalized shape.");
    TT_ASSERT(operand >= 0 && operand < 3, "Operand index out of range");

    if (operand == 2)
    {
        NodeContext grad_reduced = grad;
        if (grad.shape.size() == 3 && grad.shape[0] != 1)
        {
            // has to reduce over batch first
            grad_reduced = dc.op(
                graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{0}}, {"keep_dim", true}}), {grad});
        }
        // Gradient w.r.t. bias (beta): dbeta = reduce_sum(grad, dim=-2, keep_dim=True)
        NodeContext dbeta = dc.op(
            graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{-2}}, {"keep_dim", true}}),
            {grad_reduced});
        dc.fuse(dbeta);
        return;
    }

    // output = xhat_weighted + bias
    auto output_operands = dc.get_operands(output);
    TT_ASSERT(output_operands.size() == 2, "Expected output to have 2 operands: [xhat_weighted, bias]");
    NodeContext xhat_weighted = output_operands[0];

    // xhat_weighted = xhat * weights
    auto xhat_weighted_operands = dc.get_operands(xhat_weighted);
    TT_ASSERT(xhat_weighted_operands.size() == 2, "Expected xhat_weighted to have 2 operands: [xhat, weights]");
    NodeContext xhat = xhat_weighted_operands[0];

    // xhat = xmu * ivar
    auto xhat_operands = dc.get_operands(xhat);
    TT_ASSERT(xhat_operands.size() == 2, "Expected xhat to have 2 operands: [xmu, ivar]");
    NodeContext ivar = xhat_operands[1];

    if (operand == 1)
    {
        // Gradient w.r.t. weights (gamma): dgamma = reduce_sum(xhat * grad, dim=-2, keep_dim=True)
        NodeContext xhat_grad = dc.op(graphlib::OpType("multiply", {}, {}), {xhat, grad});
        NodeContext xhat_grad_reduced = xhat_grad;
        if (xhat_grad.shape.size() == 3 && xhat_grad.shape[0] != 1)
        {
            // has to reduce over batch first
            xhat_grad_reduced = dc.op(
                graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{0}}, {"keep_dim", true}}),
                {xhat_grad});
        }
        NodeContext dgamma = dc.op(
            graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{-2}}, {"keep_dim", true}}),
            {xhat_grad_reduced});
        dc.fuse(dgamma);
        return;
    }

    // operand == 0: Gradient w.r.t. input
    NodeContext dxhat = dc.op(graphlib::OpType("multiply", {}, {}), {grad, gamma});

    // sum_1 = reduce_sum(dxhat, dim, keep_dim=True)
    NodeContext sum_1 =
        dc.op(graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{dim}}, {"keep_dim", true}}), {dxhat});

    // sum_2 = reduce_sum(dxhat * xhat, dim, keep_dim=True)
    NodeContext dxhat_xhat = dc.op(graphlib::OpType("multiply", {}, {}), {dxhat, xhat});
    NodeContext sum_2 = dc.op(
        graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{dim}}, {"keep_dim", true}}), {dxhat_xhat});

    // xhat_sum_2 = xhat * sum_2
    NodeContext xhat_sum_2 = dc.op(graphlib::OpType("multiply", {}, {}), {xhat, sum_2});

    // sum_1_sum_2_add = sum_1 + xhat_sum_2
    NodeContext sum_1_sum_2_add = dc.op(graphlib::OpType("add", {}, {}), {sum_1, xhat_sum_2});

    // N_recip = 1.0 / N
    std::vector<uint32_t> sum_shape = sum_1_sum_2_add.shape.as_vector<uint32_t>();
    std::vector<int64_t> sum_shape_int64(sum_shape.begin(), sum_shape.end());
    at::Tensor N_recip_tensor = torch::zeros(sum_shape_int64) + (1.0f / static_cast<float>(N));
    NodeContext N_recip_node = dc.tensor(N_recip_tensor);

    // N_recip_add = N_recip * sum_1_sum_2_add
    NodeContext N_recip_add = dc.op(graphlib::OpType("multiply", {}, {}), {N_recip_node, sum_1_sum_2_add});

    // dxhat_add_sub = dxhat - N_recip_add
    NodeContext dxhat_add_sub = dc.op(graphlib::OpType("subtract", {}, {}), {dxhat, N_recip_add});

    // dx = ivar * dxhat_add_sub
    NodeContext dx = dc.op(graphlib::OpType("multiply", {}, {}), {ivar, dxhat_add_sub});

    dc.fuse(dx);
    return;
}

}  // namespace layernorm_bw
}  // namespace ops
}  // namespace tt
