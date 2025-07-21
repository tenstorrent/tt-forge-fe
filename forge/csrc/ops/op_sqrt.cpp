// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

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
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Sqrt, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Sqrt should have one input");
    TT_ASSERT(op.attrs().size() == 0, "Sqrt should not have any attributes");

    return torch::sqrt(tensors[0]);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Sqrt, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Sqrt should have one input");
    TT_ASSERT(op.attrs().size() == 0, "Sqrt should not have any attributes");

    return {Shape::create(in_shapes[0]), {}};
}

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
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

}  // namespace sqrt
}  // namespace ops
}  // namespace tt
