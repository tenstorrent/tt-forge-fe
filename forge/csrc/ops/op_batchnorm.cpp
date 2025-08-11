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

    at::Tensor input = tensors[0];
    at::Tensor weight = tensors[1];
    at::Tensor bias = tensors[2];
    at::Tensor running_mean = tensors[3];
    at::Tensor running_var = tensors[4];
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
    TT_THROW("Back propagation for Batchnorm op is not implemented yet");
    unreachable();
}

}  // namespace batchnorm
}  // namespace ops
}  // namespace tt
