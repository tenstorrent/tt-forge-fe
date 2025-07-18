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
namespace sqrt
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Sqrt, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Sqrt should have one input");
    TT_ASSERT(op.attrs().size() == 0, "Sqrt should not have any attributes");

    return torch::sqrt(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Sqrt, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Sqrt should have one input");
    TT_ASSERT(op.attrs().size() == 0, "Sqrt should not have any attributes");

    return {graphlib::Shape::create(in_shapes[0]), {}};
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Sqrt, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Sqrt should have one input");
    TT_ASSERT(operand == 0, "Invalid operand index");
    TT_ASSERT(op.attrs().size() == 0, "Sqrt should not have any attributes");

    auto constant_half = ac.autograd->create_constant(ac, 0.5f);
    auto reciprocal = ac.autograd->create_op(ac, graphlib::OpType("reciprocal"), {output});
    auto mult = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {constant_half, reciprocal});
    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {gradient, mult});
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_ASSERT(inputs.size() == 1, "Sqrt should have one input");

    auto shape_tuple = sqrt::shape(op, inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1L, std::multiplies<long>());
}

}  // namespace sqrt
}  // namespace ops
}  // namespace tt
