// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op.hpp"

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace nop
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Nop, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Nop should have one input.");

    return tensors[0];
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Nop, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Nop should have one input.");

    return {Shape::create(in_shapes[0]), {}};
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Nop, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Nop should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for nop.");

    return ac.autograd->create_op(ac, Op("nop"), {gradient});
}

}  // namespace nop
}  // namespace ops
}  // namespace tt
