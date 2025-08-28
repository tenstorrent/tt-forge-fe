// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tuple>
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
namespace broadcast
{
using namespace graphlib;

/**
 * Returns broadcasted tensor with torch broadcast.
 * Since tensor shape should be big enough to set new size on dim index, we need to expand it if it is not big
 * enough. If dim is possitive, we need to have at least dim+1 elements, and if it negative, we need -dim.
 * Negative elements are indexed as shape.size() + dim (which is negative), meaning dim's element from back.
 */
at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Broadcast, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Broadcast should have single input tensor.");
    TT_ASSERT(op.attrs().size() >= 2, "Broadcast should have at least two attributes - dim and size.");

    int dim = op.attr_as<int>("dim");
    int size = op.attr_as<int>("size");

    int64_t min_shape_size = dim >= 0 ? dim + 1 : -dim;
    at::Tensor tensor = tensors[0];

    while (tensor.dim() < min_shape_size) tensor = tensor.unsqueeze(0);

    std::vector<int64_t> target_shape = tensor.sizes().vec();

    size_t idx = dim >= 0 ? dim : target_shape.size() + dim;
    target_shape[idx] = size;

    return tensor.broadcast_to(target_shape);
}

/**
 * Returns shape based on provided input shape.
 * Since tensor shape should be big enough to set new size on dim index, we need to expand it if it is not big
 * enough. If dim is possitive, we need to have at least dim+1 elements, and if it negative, we need -dim.
 * Negative elements are indexed as shape.size() + dim (which is negative), meaning dim's element from back.
 */
std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Broadcast, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Broadcast should have single input shape.");
    TT_ASSERT(op.attrs().size() >= 2, "Broadcast should have at least two attributes - dim and size.");

    int dim = op.attr_as<int>("dim");
    int size = op.attr_as<int>("size");

    size_t min_shape_size = dim >= 0 ? dim + 1 : -dim;
    std::vector<std::uint32_t> target_shape = in_shapes[0];

    while (target_shape.size() < min_shape_size) target_shape.insert(target_shape.begin(), 1);

    size_t idx = dim >= 0 ? dim : target_shape.size() + dim;
    target_shape[idx] = size;

    return {Shape::create(target_shape), {}};
}

tt::graphlib::NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Broadcast, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Broadcast should have single input.");
    TT_ASSERT(op.attrs().size() >= 2, "Broadcast should have at least two attributes - dim and size.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    int dim = op.attr_as<int>("dim");

    return ac.autograd->create_op(
        ac, Op("reduce_sum", {{"dim_arg", std::vector<int>({dim})}, {"keep_dim", true}}), {gradient});
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Broadcast, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Broadcast should have single input.");
    TT_ASSERT(op.attrs().size() >= 2, "Broadcast should have at least two attributes - dim and size.");

    if (op.attr_as<int>("size") == 1)
        dc.fuse(dc.op(Op("nop"), {inputs[0]}));
}

}  // namespace broadcast
}  // namespace ops
}  // namespace tt
