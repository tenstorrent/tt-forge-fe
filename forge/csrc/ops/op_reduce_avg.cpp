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
namespace reduce_avg
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "reduce_avg should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_avg should have 2 attrs (dim, keep_dim).");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    bool keep_dim = op.attr_as<bool>("keep_dim");

    return torch::mean(tensors[0], dim, keep_dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "reduce_avg should have single input shape.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_avg should have 2 attrs (dim, keep_dim).");

    return op_common::reduce_ops_shape(op, in_shapes);
}

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_avg should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    // For avg, gradient needs to be broadcast back to original shape
    // with scale factor 1 / size
    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    if (dim < 0)
        dim += inputs[0].shape.size();
    std::uint32_t size = inputs[0].shape[dim];

    NodeContext unsqueeze = gradient;
    if (!op.attr_as<bool>("keep_dim"))
    {
        // If keep_dim is false, we need to unsqueeze the gradient to match the input shape.
        unsqueeze = ac.autograd->create_op(ac, graphlib::OpType("unsqueeze", {}, {{"dim", dim}}), {gradient});
    }

    std::vector<int> repeats(unsqueeze.shape.size(), 1);
    repeats[dim] = size;
    NodeContext repeat =
        ac.autograd->create_op(ac, graphlib::OpType("repeat", {}, {{"repeats", repeats}}), {unsqueeze});

    NodeContext consts = ac.autograd->create_constant(ac, 1.0 / size);

    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {repeat, consts});
}

void decompose_initial(
    const graphlib::OpType &old_op_type,
    const Op &op,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceAvg, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_avg should have single input.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    if (dim < 0)
        dim += inputs[0].shape.size();

    if (inputs[0].shape[dim] == 1)
    {
        // We are reducing on a dimension that is already 1, which is potentially a no-op.
        if (op.attr_as<bool>("keep_dim"))
        {
            // `keep_dim` is true, hence we don't need to do anything.
            NodeContext result = dc.op(graphlib::OpType("nop"), {inputs[0]});
            dc.fuse(result);
            return;
        }

        // In this case, we can replace `reduce_sum` with a `squeeze` operation.
        NodeContext result = dc.op(graphlib::OpType("squeeze", {}, {{"dim", dim}}), {inputs[0]});
        dc.fuse(result);
    }
}

}  // namespace reduce_avg
}  // namespace ops
}  // namespace tt
