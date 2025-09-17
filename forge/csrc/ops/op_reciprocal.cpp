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
namespace reciprocal
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Reciprocal, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "OpReciprocal::eval should have single input tensor.");
    // Add epsilon to avoid infinity
    return torch::reciprocal(tensors[0] + 1e-10);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Reciprocal, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "OpReciprocal::shape should have single input shape.");
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
    /**
     * Backward implementation:
     * dx = -1/x^2 * grad
     */

    TT_DBG_ASSERT(op.type() == OpType::Reciprocal, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Reciprocal should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    auto sq = ac.autograd->create_op(ac, Op(OpType::Multiply), {output, output});
    auto neg_one = ac.autograd->create_constant(ac, -1.0);
    auto neg = ac.autograd->create_op(ac, Op(OpType::Multiply), {sq, neg_one});

    return ac.autograd->create_op(ac, Op(OpType::Multiply), {neg, gradient});
}

}  // namespace reciprocal
}  // namespace ops
}  // namespace tt
