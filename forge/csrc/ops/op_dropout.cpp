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
namespace dropout
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Dropout, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Dropout should have one input.");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed.");

    torch::manual_seed(op.attr_as<int>("seed"));
    return torch::dropout(tensors[0], op.attr_as<float>("p"), op.attr_as<bool>("training"));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Dropout, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Dropout should have one input.");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed.");

    return {Shape::create(in_shapes[0]), {}};
};

NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Dropout, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Dropout should have one input.");
    TT_ASSERT(op.attrs().size() == 3, "Dropout should have 3 attributes: p, training, seed.");
    TT_ASSERT(operand == 0, "Invalid operand index for dropout.");

    // Apply dropout to gradient with the same parameters.
    return ac.autograd->create_op(ac, graphlib::OpType("dropout", {}, op.attrs()), {gradient});
}

}  // namespace dropout
}  // namespace ops
}  // namespace tt
