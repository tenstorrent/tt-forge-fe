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
namespace downsample_2d
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Downsample2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Downsample2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Downsample2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Downsample2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

void decompose_post_optimize(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Downsample2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Downsample2d, "Wrong op type.");
    TT_THROW("Not implemented.");
    unreachable();
}

}  // namespace downsample_2d
}  // namespace ops
}  // namespace tt
