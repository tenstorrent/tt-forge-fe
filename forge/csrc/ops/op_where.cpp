// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "ops/op_common.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace where
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Where, "Wrong op type.");
    TT_ASSERT(tensors.size() == 3, "Where should have three operands: condition, x, y.");

    at::Tensor condition = tensors[0].to(torch::kBool);
    return torch::where(condition, tensors[1], tensors[2]);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Where, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 3, "Where should have three operands: condition, x, y.");

    return op_common::eltwise_nary_shape(in_shapes);
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Where, "Wrong op type.");
    TT_THROW("Where op does not have backward implemented.");
    unreachable();
}

}  // namespace where
}  // namespace ops
}  // namespace tt
