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
namespace batchnorm
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Batchnorm, "Wrong op type.");
    TT_ASSERT(tensors.size() == 5, "Batchnorm should have five operands.");

    const at::Tensor &input = tensors[0];
    const at::Tensor &weight = tensors[1];
    const at::Tensor &bias = tensors[2];
    const at::Tensor &running_mean = tensors[3];
    const at::Tensor &running_var = tensors[4];
    double epsilon = op.attr_as<float>("epsilon");

    // Clone running statistics without gradients (batch_norm requires no gradients for these)
    at::Tensor running_mean_no_grad = running_mean.clone().detach();
    at::Tensor running_var_no_grad = running_var.clone().detach();

    return torch::nn::functional::batch_norm(
        input,
        running_mean_no_grad,
        running_var_no_grad,
        torch::nn::functional::BatchNormFuncOptions().weight(weight).bias(bias).training(false).momentum(0.0).eps(
            epsilon));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Batchnorm, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 5, "Batchnorm should have five input shapes.");

    // Batchnorm output shape is the same as the input shape
    return std::make_tuple(Shape::create(in_shapes[0]), std::vector<DimBroadcast>{});
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
    TT_DBG_ASSERT(op.type() == OpType::Batchnorm, "Wrong op type.");
    throw std::runtime_error("Back propagation for Batchnorm op is not implemented yet");
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Batchnorm, "Wrong op type.");
    TT_ASSERT(inputs.size() == 5, "Batchnorm should have five operands.");

    const NodeContext &input = inputs[0];
    const NodeContext &weight = inputs[1];
    const NodeContext &bias = inputs[2];
    const NodeContext &running_mean = inputs[3];
    const NodeContext &running_var = inputs[4];
    float epsilon = op.attr_as<float>("epsilon");

    // Create constant tensors
    std::vector<long> mean_var_shape;
    for (int dim : running_var.shape.as_vector<int>())
    {
        mean_var_shape.push_back(static_cast<long>(dim));
    }
    std::vector<long> mean_shape;
    for (int dim : running_mean.shape.as_vector<int>())
    {
        mean_shape.push_back(static_cast<long>(dim));
    }
    auto epsilon_tensor = dc.tensor(torch::zeros(mean_var_shape) + epsilon);
    auto neg_one = dc.tensor(torch::zeros(mean_shape) - 1.0);

    // Decompose: output = weight * (input - running_mean) / sqrt(running_var + epsilon) + bias
    auto var_eps = dc.op(graphlib::OpType("add", {}, {}), {running_var, epsilon_tensor});
    auto sqrt_result = dc.op(graphlib::OpType("sqrt", {}, {}), {var_eps});
    auto reciprocal_result = dc.op(graphlib::OpType("reciprocal", {}, {}), {sqrt_result});
    auto weighted = dc.op(graphlib::OpType("multiply", {}, {}), {reciprocal_result, weight});
    auto neg_mean = dc.op(graphlib::OpType("multiply", {}, {}), {neg_one, running_mean});
    auto weighted_mean = dc.op(graphlib::OpType("multiply", {}, {}), {weighted, neg_mean});
    auto weighted_bias = dc.op(graphlib::OpType("add", {}, {}), {weighted_mean, bias});

    // Unsqueeze to match input dimensions
    auto weighted_bias_unsqueezed = dc.op(graphlib::OpType("unsqueeze", {}, {{"dim", 1}}), {weighted_bias});
    weighted_bias_unsqueezed = dc.op(graphlib::OpType("unsqueeze", {}, {{"dim", 1}}), {weighted_bias_unsqueezed});

    auto weighted_var_unsqueezed = dc.op(graphlib::OpType("unsqueeze", {}, {{"dim", 1}}), {weighted});
    weighted_var_unsqueezed = dc.op(graphlib::OpType("unsqueeze", {}, {{"dim", 1}}), {weighted_var_unsqueezed});

    auto scaled = dc.op(graphlib::OpType("multiply", {}, {}), {input, weighted_var_unsqueezed});
    auto result = dc.op(graphlib::OpType("add", {}, {}), {scaled, weighted_bias_unsqueezed});

    dc.fuse(result);
}

}  // namespace batchnorm
}  // namespace ops
}  // namespace tt
