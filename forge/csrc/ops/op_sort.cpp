// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <tuple>
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
namespace sort
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Sort, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Sort should have one input");

    int64_t axis = -1;
    if (op.has_attr("axis"))
        axis = op.attr_as<int>("axis");

    bool is_ascend = true;
    if (op.has_attr("is_ascend"))
        is_ascend = op.attr_as<bool>("is_ascend");

    bool descending = !is_ascend;

    auto result = at::sort(tensors[0], axis, descending);
    return std::get<0>(result);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Sort, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Sort should have one input");

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
    TT_DBG_ASSERT(op.type() == OpType::Sort, "Wrong op type.");
    TT_ASSERT(false, "Sort does not have backward.");
    unreachable();
}

}  // namespace sort
}  // namespace ops
}  // namespace tt
