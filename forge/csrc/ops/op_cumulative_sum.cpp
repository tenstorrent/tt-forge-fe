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
namespace cumulative_sum
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::CumulativeSum, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Cumulative sum should have one input.");
    TT_ASSERT(op.attrs().size() == 1, "Cumulative sum should have one attribute.");

    return torch::cumsum(tensors[0], op.attr_as<int>("dim"));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::CumulativeSum, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Cumulative sum should have one input.");
    TT_ASSERT(op.attrs().size() == 1, "Cumulative sum should have one attribute.");

    return {Shape::create(in_shapes[0]), {}};
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
    TT_DBG_ASSERT(op.type() == OpType::CumulativeSum, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Cumulative sum should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for cumsum.");
    TT_ASSERT(op.has_attr("dim"), "Cumulative sum should have one attribute.");

    TT_ASSERT(false, "Cumsum does not have backward implemented.");
    unreachable();
}

}  // namespace cumulative_sum
}  // namespace ops
}  // namespace tt
