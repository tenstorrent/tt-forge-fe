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
namespace adaptive_max_pool_2d
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::AdaptiveMaxPool2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::AdaptiveMaxPool2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
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
    TT_DBG_ASSERT(op.type() == OpType::AdaptiveMaxPool2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::AdaptiveMaxPool2d, "Wrong op type.");
    TT_THROW("Not implemented.");
}

void decompose_post_optimize(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::AdaptiveMaxPool2d, "Wrong op type.");
    TT_THROW("Not implemented.");
}

void decompose_post_autograd(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::AdaptiveMaxPool2d, "Wrong op type.");
    TT_THROW("Not implemented.");
}

}  // namespace adaptive_max_pool_2d
}  // namespace ops
}  // namespace tt
