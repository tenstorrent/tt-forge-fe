// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
namespace reduce_sum
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "reduce_sum should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_sum should have 2 attrs (dim, keep_dim).");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    bool keep_dim = op.attr_as<bool>("keep_dim");

    return torch::sum(tensors[0], dim, keep_dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "reduce_sum should have single input shape.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_sum should have 2 attrs (dim, keep_dim).");

    return op_common::reduce_ops_shape(op, in_shapes);
}

tt::graphlib::NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_sum should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    std::uint32_t size = inputs[0].shape[dim];

    std::optional<NodeContext> unsqueeze;
    if (!op.attr_as<bool>("keep_dim"))
    {
        // If `keep_dim` is false, we need to unsqueeze the gradient to match the input shape.
        unsqueeze = ac.autograd->create_op(ac, Op(OpType::Unsqueeze, {{"dim", dim}}), {gradient});
    }

    // For sum, gradient just needs to be broadcast back to original shape.
    NodeContext broadcast = ac.autograd->create_op(
        ac,
        Op(OpType::Broadcast, {{"dim", dim}, {"size", static_cast<int>(size)}}),
        {unsqueeze.has_value() ? *unsqueeze : gradient});
    return broadcast;
}

void decompose_initial(

    const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_sum should have single input.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];

    if (inputs[0].shape[dim] == 1)
    {
        // We are reducing on a dimension that is already 1, which is potentially a no-op.
        if (op.attr_as<bool>("keep_dim"))
        {
            // `keep_dim` is true, hence we don't need to do anything.
            NodeContext result = dc.op(Op(OpType::Nop), {inputs[0]});
            dc.fuse(result);
            return;
        }

        // In this case, we can replace `reduce_sum` with a `squeeze` operation.
        NodeContext result = dc.op(Op(OpType::Squeeze, {{"dim", dim}}), {inputs[0]});
        dc.fuse(result);
    }
}

}  // namespace reduce_sum
}  // namespace ops
}  // namespace tt
