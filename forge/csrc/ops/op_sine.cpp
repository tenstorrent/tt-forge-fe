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
namespace sine
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Sine, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "sine::eval should have single input tensor.");
    return torch::sin(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Sine, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "sine::shape should have single input shape.");
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    /**
     * Derivative of sin(x) is cos(x)
     * So backward pass: cos(inputs[0]) * grad
     */

    TT_DBG_ASSERT(op.type() == OpType::Sine, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Sine should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    graphlib::NodeContext cosine_output = ac.autograd->create_op(ac, graphlib::OpType("cosine"), {inputs[0]});
    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {cosine_output, gradient});
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Sine, "Wrong op type.");

    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape_tuple = sine::shape(op, inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace sine
}  // namespace ops
}  // namespace tt
