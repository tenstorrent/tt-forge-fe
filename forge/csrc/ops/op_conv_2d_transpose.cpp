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
namespace conv_2d_transpose
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2dTranspose, "Wrong op type.");

    // Implementation.

    return {};
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2dTranspose, "Wrong op type.");

    // Implementation.

    return {};
}

NodeContext backward(
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2dTranspose, "Wrong op type.");

    // Implementation.

    return {gradient};
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2dTranspose, "Wrong op type.");

    // Implementation.
}

void decompose_post_optimize(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2dTranspose, "Wrong op type.");

    // Implementation.
}

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2dTranspose, "Wrong op type.");

    // Implementation.
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Conv2dTranspose, "Wrong op type.");

    // Implementation.

    return 0;
}

}  // namespace conv_2d_transpose
}  // namespace ops
}  // namespace tt
