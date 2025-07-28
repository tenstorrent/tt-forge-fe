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
namespace repeat_interleave
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    return op.base_eval(old_op_type, tensors);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    return op.base_shape(old_op_type, in_shapes);
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
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    return op.base_backward(old_op_type, ac, operand, inputs, output, gradient);
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    return op.base_decompose(old_op_type, "get_f_forge_decompose", dc, inputs);
}

void decompose_post_optimize(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    return op.base_decompose(old_op_type, "get_f_forge_decompose_post_optimize", dc, inputs);
}

void decompose_post_autograd(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    return op.base_decompose(old_op_type, "get_f_forge_decompose_post_autograd", dc, inputs);
}

long initial_flops_estimate(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    return op.base_initial_flops_estimate(old_op_type, inputs);
}

}  // namespace repeat_interleave
}  // namespace ops
}  // namespace tt
