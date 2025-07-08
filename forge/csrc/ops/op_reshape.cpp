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
namespace reshape
{

using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Reshape, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Reshape should have single input tensor.");

    const std::vector<int> &shape_attr = op.attr_as<std::vector<int>>("shape");
    return tensors[0].reshape(std::vector<int64_t>(shape_attr.begin(), shape_attr.end()));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Reshape, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Reshape should have single input shape.");

    return std::make_tuple(Shape::create(op.attr_as<std::vector<int>>("shape")), std::vector<DimBroadcast>{});
}

tt::graphlib::NodeContext backward(
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Reshape, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Reshape should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    return ac.autograd->create_op(
        ac, graphlib::OpType("reshape", {}, {{"shape", inputs[0].shape.as_vector<int>()}}), {gradient});
}

void decompose_reshape(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Reshape, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Reshape should have single input.");

    std::vector<int> input_shape = inputs[0].shape.as_vector<int>();
    const std::vector<int> &shape = op.attr_as<std::vector<int>>("shape");

    if (shape == input_shape)
        return;

    std::vector<int> new_shape = shape;

    int rank = 0;
    for (; new_shape.size() < input_shape.size(); --rank) new_shape.insert(new_shape.begin(), 1);
    for (; new_shape.size() > input_shape.size() && new_shape[0] == 1; ++rank) new_shape.erase(new_shape.begin());

    if (new_shape != input_shape || rank == 0)
        return;

    NodeContext result = inputs[0];  // clang-format off
    for (; rank < 0; ++rank) result = dc.op(graphlib::OpType("squeeze",   {0},                           {{"dim", 0}}), {std::move(result)});
    for (; rank > 0; --rank) result = dc.op(graphlib::OpType("unsqueeze", {0, int(result.shape.size())}, {{"dim", 0}}), {result});  // clang-format on

    dc.fuse(result);
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    decompose_reshape(op, dc, inputs);
}

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    decompose_reshape(op, dc, inputs);
}

}  // namespace reshape
}  // namespace ops
}  // namespace tt
