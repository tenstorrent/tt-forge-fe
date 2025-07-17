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
namespace cosine
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Cosine, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "cosine::eval should have single input tensor.");
    return torch::cos(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Cosine, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "cosine::shape should have single input shape.");
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
    /**
     * Derivative of cos(x) is -sin(x)
     * So backward pass: -sin(inputs[0]) * grad
     */

    TT_DBG_ASSERT(op.type() == OpType::Cosine, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Cosine should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    graphlib::NodeContext sine_output = ac.autograd->create_op(ac, graphlib::OpType("sine"), {inputs[0]});
    graphlib::NodeContext neg_sine =
        ac.autograd->create_op(ac, graphlib::OpType("multiply"), {sine_output, ac.autograd->create_constant(ac, -1.0)});
    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {neg_sine, gradient});
}

long initial_flops_estimate(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Cosine, "Wrong op type.");

    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape_tuple =
        cosine::shape(old_op_type, op, inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace cosine
}  // namespace ops
}  // namespace tt
