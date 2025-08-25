// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace index_copy
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::IndexCopy, "Wrong op type.");
    TT_DBG_ASSERT(tensors.size() == 3, "IndexCopy expects 3 tensors: input, index, src");

    int dim = op.attr_as<int>("dim");

    std::vector<at::Tensor> promoted_tensors = op_common::promote_floating_dtypes(tensors);
    at::Tensor index_long = promoted_tensors[1].to(torch::kLong);

    return promoted_tensors[0].index_copy(dim, index_long, promoted_tensors[2]);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::IndexCopy, "Wrong op type.");
    TT_DBG_ASSERT(in_shapes.size() == 3, "IndexCopy expects 3 input shapes: input, index, src");

    return std::make_tuple(Shape::create(in_shapes[0]), std::vector<DimBroadcast>{});
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
    TT_DBG_ASSERT(op.type() == OpType::IndexCopy, "Wrong op type.");
    TT_THROW("IndexCopy backward is not implemented");
    unreachable();
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::IndexCopy, "Wrong op type.");
    TT_DBG_ASSERT(inputs.size() == 3, "IndexCopy expects 3 inputs: operandA, index, value");

    int dim = op.attr_as<int>("dim");
    const NodeContext &operandA = inputs[0];
    const NodeContext &index = inputs[1];
    const NodeContext &value = inputs[2];

    if (dim > 0)
        dim -= operandA.shape.size();

    // Check if this is the special case for decomposition to FillCache/UpdateCache
    if (dim == -2 && operandA.shape.size() == 4 && value.shape.size() == 4)
    {
        if (index.shape[0] > 1)
        {
            auto result = dc.op(graphlib::OpType("fill_cache", {{"batch_offset", 0}}), {operandA, value});
            dc.fuse(result);
        }
        else
        {
            auto result = dc.op(graphlib::OpType("update_cache", {{"batch_offset", 0}}), {operandA, value, index});
            dc.fuse(result);
        }
    }
}

}  // namespace index_copy
}  // namespace ops
}  // namespace tt
