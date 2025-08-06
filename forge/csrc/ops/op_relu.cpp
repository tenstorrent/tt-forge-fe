// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace relu
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Relu, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "OpRelu::eval should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 0, "OpRelu::eval should have no attributes.");

    return torch::relu(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Relu, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "OpRelu::shape should have single input shape.");
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Relu, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Relu should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    // Standard ReLU gradient: max(0, x)
    tt::graphlib::NodeContext zero_constant = ac.autograd->create_constant(ac, 0.0f);
    tt::graphlib::NodeContext relu_derivative =
        ac.autograd->create_op(ac, graphlib::OpType("greater_equal"), {inputs[0], zero_constant});

    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {relu_derivative, gradient});
}

}  // namespace relu
}  // namespace ops
}  // namespace tt
