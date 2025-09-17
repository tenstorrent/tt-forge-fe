// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace heaviside
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 2, "Heaviside should have two input tensors.");
    return torch::heaviside(tensors[0], tensors[1]);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Heaviside, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "Heaviside should have two input shapes.");
    TT_ASSERT(op.attrs().size() == 0, "Heaviside should not have any attrs.");

    return op_common::compute_elementwise_binary_shape(in_shapes);
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_ASSERT(false, "Heaviside does not have backward.");
    unreachable();
}

/**
 * Decompose Heaviside: result = (x > 0) + (x == 0) * y
 */
void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Heaviside, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "Heaviside should have two inputs");

    auto x = inputs[0];
    auto y = inputs[1];

    auto zero = dc.op(Op(OpType::Constant, {{"c", 0.0f}}), {});
    auto x_gt = dc.op(Op(OpType::Greater), {x, zero});
    auto x_eq = dc.op(Op(OpType::Equal), {x, zero});
    auto res = dc.op(Op(OpType::Multiply), {x_eq, y});
    res = dc.op(Op(OpType::Add), {res, x_gt});

    dc.fuse(res, 0);
}

}  // namespace heaviside
}  // namespace ops
}  // namespace tt
