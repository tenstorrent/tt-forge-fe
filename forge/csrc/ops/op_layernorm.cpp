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
namespace layernorm
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Layernorm, "Wrong op type.");
    TT_ASSERT(tensors.size() == 3, "Layernorm should have three operands.");

    at::Tensor input = tensors[0];  // Input tensor
    at::Tensor gamma = tensors[1];  // weights, weight re-scaling parameter
    at::Tensor beta = tensors[2];   // bias, weight re-centering parameter

    int dim = op.attr_as<int>("dim");
    float epsilon = op.attr_as<float>("epsilon");

    TT_ASSERT(
        dim == -1 || dim == static_cast<int>(input.dim()) - 1,
        "Normalization can be done only over the last dimension");
    TT_ASSERT(gamma.size(-1) == input.size(-1), "Weights shape must be the same as normalized shape.");
    TT_ASSERT(beta.size(-1) == input.size(-1), "Bias shape must be the same as normalized shape.");

    // Check that all dimensions except the last one are 1 for gamma and beta
    for (int i = 0; i < gamma.dim() - 1; ++i)
    {
        TT_ASSERT(gamma.size(i) == 1, "All dimensions but the last one must be 1 for gamma");
    }
    for (int i = 0; i < beta.dim() - 1; ++i)
    {
        TT_ASSERT(beta.size(i) == 1, "All dimensions but the last one must be 1 for beta");
    }

    // Get the normalized shape (last dimension)
    std::vector<int64_t> normalized_shape = {input.size(-1)};

    // Reshape gamma and beta to match the normalized shape
    at::Tensor gamma_reshaped = gamma.reshape({gamma.size(-1)});
    at::Tensor beta_reshaped = beta.reshape({beta.size(-1)});

    // Ensure all tensors have the same dtype
    at::ScalarType target_dtype = input.scalar_type();
    gamma_reshaped = gamma_reshaped.to(target_dtype);
    beta_reshaped = beta_reshaped.to(target_dtype);

    return torch::nn::functional::layer_norm(
        input,
        torch::nn::functional::LayerNormFuncOptions(normalized_shape)
            .weight(gamma_reshaped)
            .bias(beta_reshaped)
            .eps(epsilon));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Layernorm, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 3, "Layernorm should have three operands.");

    // LayerNorm output shape is the same as input shape
    return std::make_tuple(Shape::create(in_shapes[0]), std::vector<DimBroadcast>{});
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Layernorm, "Wrong op type.");
    TT_ASSERT(inputs.size() == 3, "Layernorm should have three operands.");
    TT_ASSERT(operand >= 0 && operand < static_cast<int>(inputs.size()), "Operand index out of the input range.");

    int dim = op.attr_as<int>("dim");
    float epsilon = op.attr_as<float>("epsilon");

    // Create layernorm_bw op with operand index
    std::vector<NodeContext> bw_inputs = {inputs[0], inputs[1], inputs[2], gradient, output};

    return ac.autograd->create_op(
        ac, Op(OpType::LayernormBw, {{"dim", dim}, {"epsilon", epsilon}, {"operand", operand}}), bw_inputs);
}

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Layernorm, "Wrong op type.");
    TT_ASSERT(inputs.size() == 3, "Layernorm should have three operands.");

    NodeContext input = inputs[0];
    NodeContext weights = inputs[1];
    NodeContext bias = inputs[2];

    int dim = op.attr_as<int>("dim");
    float epsilon = op.attr_as<float>("epsilon");

    // Normalize negative dimension
    if (dim < 0)
    {
        dim += static_cast<int>(input.shape.size());
    }

    std::vector<uint32_t> input_shape = input.shape.as_vector<uint32_t>();
    std::vector<uint32_t> gamma_shape = weights.shape.as_vector<uint32_t>();
    std::vector<uint32_t> beta_shape = bias.shape.as_vector<uint32_t>();

    TT_ASSERT(
        dim == static_cast<int>(input_shape.size()) - 1, "Normalization can be done only over the last dimension.");
    TT_ASSERT(gamma_shape.back() == input_shape.back(), "Weights shape must be the same as normalized shape.");
    TT_ASSERT(beta_shape.back() == input_shape.back(), "Bias shape must be the same as normalized shape.");

    // Calculate mean: mu = sum(input, dim) / N
    NodeContext mu = dc.op(Op(OpType::ReduceSum, {{"dim_arg", std::vector<int>{dim}}, {"keep_dim", true}}), {input});

    // Create tensor for division by N
    std::vector<int64_t> input_shape_int64(input_shape.begin(), input_shape.end());
    at::Tensor divider_tensor = torch::zeros(input_shape_int64) + (1.0f / static_cast<float>(input_shape[dim]));
    NodeContext divider = dc.tensor(divider_tensor);
    mu = dc.op(Op(OpType::Multiply), {divider, mu});

    // Calculate xmu = input - mu
    NodeContext xmu = dc.op(Op(OpType::Subtract), {input, mu});

    // Calculate squared difference: sq = xmu * xmu
    NodeContext sq = dc.op(Op(OpType::Multiply), {xmu, xmu});

    // Calculate variance: var = sum(sq, dim) / N
    NodeContext var = dc.op(Op(OpType::ReduceSum, {{"dim_arg", std::vector<int>{dim}}, {"keep_dim", true}}), {sq});
    std::vector<int64_t> var_shape = var.shape.as_vector<int64_t>();
    at::Tensor var_divider_tensor = torch::zeros(var_shape) + (1.0f / static_cast<float>(input_shape[dim]));
    NodeContext var_divider = dc.tensor(var_divider_tensor);
    var = dc.op(Op(OpType::Multiply), {var_divider, var});

    // Add epsilon: var_add = var + epsilon
    at::Tensor epsilon_tensor = torch::zeros(var_shape) + epsilon;
    NodeContext epsilon_node = dc.tensor(epsilon_tensor);
    NodeContext var_add = dc.op(Op(OpType::Add), {var, epsilon_node});

    // Calculate standard deviation: std = sqrt(var_add)
    NodeContext std = dc.op(Op(OpType::Sqrt), {var_add});

    // Calculate inverse variance: ivar = 1 / std
    NodeContext ivar = dc.op(Op(OpType::Reciprocal), {std});

    // Normalize: xhat = xmu * ivar
    NodeContext xhat = dc.op(Op(OpType::Multiply), {xmu, ivar});

    // Apply weights: xhat_weighted = xhat * weights
    NodeContext xhat_weighted = dc.op(Op(OpType::Multiply), {xhat, weights});

    // Apply bias: result = xhat_weighted + bias
    NodeContext result = dc.op(Op(OpType::Add), {xhat_weighted, bias});

    dc.fuse(result);
}

}  // namespace layernorm
}  // namespace ops
}  // namespace tt
