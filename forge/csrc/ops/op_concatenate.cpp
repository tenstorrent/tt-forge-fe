// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace concatenate
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Concatenate, "Wrong op type.");
    TT_ASSERT(tensors.size() >= 1, "concatenate::eval should have at least one input tensors.");
    TT_ASSERT(op.attrs().size() == 1, "concatenate::eval should have 1 attr.");

    int dim = op.attr_as<int>("dim");

    return torch::cat(tensors, dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Concatenate, "Wrong op type.");
    TT_ASSERT(in_shapes.size() >= 1, "concatenate::shape should have at least one input shapes.");
    TT_ASSERT(op.attrs().size() == 1, "concatenate::shape should have 1 attr");

    int dim = op.attr_as<int>("dim");
    if (dim < 0)
        dim += in_shapes[0].size();

    // Shape compatibility validation for concatenation
    size_t expected_rank = in_shapes[0].size();

    for (size_t i = 1; i < in_shapes.size(); i++)
    {
        // Check if all shapes have the same rank (number of dimensions)
        TT_ASSERT(
            in_shapes[i].size() == expected_rank,
            "concatenate::shape All input shapes must have the same number of dimensions");

        // Check if all dimensions except concatenation dim are identical
        for (size_t d = 0; d < expected_rank; d++)
        {
            if (static_cast<int>(d) == dim)
                continue;

            TT_ASSERT(
                in_shapes[i][d] == in_shapes[0][d],
                "concatenate::shape All dimensions except concatenation dimension must be identical");
        }
    }

    std::vector<std::uint32_t> output_shape = in_shapes[0];

    // Concatenate along the specified dim
    for (size_t i = 1; i < in_shapes.size(); i++)
    {
        output_shape[dim] += in_shapes[i][dim];
    }

    return std::make_tuple(graphlib::Shape::create(output_shape), std::vector<graphlib::DimBroadcast>{});
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
    TT_DBG_ASSERT(op.type() == OpType::Concatenate, "Wrong op type.");
    TT_ASSERT(operand >= 0 && operand < static_cast<int>(inputs.size()), "concatenate::backward Invalid operand index");

    int dim = op.attr_as<int>("dim");
    if (dim < 0)
        dim += inputs[0].shape.size();

    uint32_t begin = 0;
    for (int i = 0; i < operand; i++)
    {
        begin += inputs[i].shape[dim];
    }

    // Create index operation to extract the slice for this operand
    uint32_t stop = begin + inputs[operand].shape[dim];
    graphlib::OpType index_op("index");
    index_op.set_attr("dim", dim);
    index_op.set_attr("start", static_cast<int>(begin));
    index_op.set_attr("stop", static_cast<int>(stop));
    index_op.set_attr("stride", 1);

    return ac.autograd->create_op(ac, index_op, {gradient});
}

void decompose_initial(
    const graphlib::OpType &old_op_type,
    const Op &op,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Concatenate, "Wrong op type.");

    if (inputs.size() == 1)
    {
        NodeContext result = dc.op(graphlib::OpType("nop"), {inputs[0]});
        dc.fuse(result);
    }
}

}  // namespace concatenate
}  // namespace ops
}  // namespace tt
