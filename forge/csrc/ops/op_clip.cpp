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
namespace clip
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Clip, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Clip should have one input.");
    TT_ASSERT(op.attrs().size() == 2, "Clip should have 2 attributes: min and max.");

    return torch::clamp(tensors[0], op.attr_as<float>("min"), op.attr_as<float>("max"));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Clip, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Clip should have one input.");
    TT_ASSERT(op.attrs().size() == 2, "Clip should have 2 attributes: min and max.");

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
    TT_DBG_ASSERT(op.type() == OpType::Clip, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Clip should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for clip.");
    TT_ASSERT(op.attrs().size() == 2, "Clip should have 2 attributes: min and max.");

    float min_val = op.attr_as<float>("min");
    float max_val = op.attr_as<float>("max");

    auto x = inputs[0];

    // Create constant scalars for min and max values
    auto min_constant = ac.autograd->create_constant(ac, min_val);
    auto max_constant = ac.autograd->create_constant(ac, max_val);

    // Create mask: (x >= min) * (x <= max)
    auto ge_min = ac.autograd->create_op(ac, graphlib::OpType("greater_equal"), {x, min_constant});
    auto le_max = ac.autograd->create_op(ac, graphlib::OpType("less_equal"), {x, max_constant});
    auto mask = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {ge_min, le_max});

    // Apply mask to gradient
    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {mask, gradient});
}

}  // namespace clip
}  // namespace ops
}  // namespace tt
