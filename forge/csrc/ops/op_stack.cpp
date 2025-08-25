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
namespace stack
{
using namespace graphlib;

static void validate_input_shapes(
    const std::vector<std::uint32_t> &first_shape, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    for (size_t i = 1; i < in_shapes.size(); i++)
    {
        TT_ASSERT(
            in_shapes[i].size() == first_shape.size(),
            "All input shapes must have the same number of dimensions for stack operation.");
        for (size_t j = 0; j < first_shape.size(); j++)
        {
            TT_ASSERT(in_shapes[i][j] == first_shape[j], "All input shapes must be identical for stack operation.");
        }
    }
}

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Stack, "Wrong op type.");
    TT_ASSERT(tensors.size() >= 1, "Stack should have at least one input tensor.");
    TT_ASSERT(op.attrs().size() == 1, "Stack should have 1 attr.");

    int dim = op.attr_as<int>("dim");
    return torch::stack(tensors, dim);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Stack, "Wrong op type.");
    TT_ASSERT(in_shapes.size() >= 1, "Stack should have at least one input shape.");
    TT_ASSERT(op.attrs().size() == 1, "Stack should have 1 attr.");

    int dim = op.attr_as<int>("dim");

    // Validate all input shapes are the same
    const auto &first_shape = in_shapes[0];
    validate_input_shapes(first_shape, in_shapes);

    // Calculate output shape by inserting a new dimension
    std::vector<std::uint32_t> output_shape = first_shape;
    int input_rank = static_cast<int>(first_shape.size());

    // For negative dim, insert at position (input_rank + dim + 1)
    if (dim < 0)
        dim = input_rank + dim + 1;

    TT_ASSERT(
        dim >= 0 && dim <= input_rank,
        "Invalid dimension for stack operation. dim=" + std::to_string(dim) + ", limits=[0, " +
            std::to_string(input_rank) + "]");
    // Insert at position dim, value is the number of inputs
    output_shape.insert(output_shape.begin() + dim, in_shapes.size());

    return std::make_tuple(Shape::create(output_shape), std::vector<DimBroadcast>());
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
    TT_DBG_ASSERT(op.type() == OpType::Stack, "Wrong op type.");
    TT_ASSERT(false, "Stack op should've been decomposed, so we don't need a backward pass.");

    return nullptr;
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Stack, "Wrong op type.");
    TT_ASSERT(op.attrs().size() == 1, "Stack should have 1 attr.");
    TT_ASSERT(inputs.size() >= 1, "Stack should have at least one input.");

    int dim = op.attr_as<int>("dim");

    // Stack operation can be decomposed as:
    // 1. Unsqueeze each input to add a dimension of size 1 at the stack dim
    // 2. Concatenate along that dimension

    std::vector<NodeContext> unsqueezed_inputs;
    int input_rank = static_cast<int>(inputs[0].shape.size());

    // For negative dim, convert to positive index
    if (dim < 0)
        dim = input_rank + dim + 1;

    TT_ASSERT(
        dim >= 0 && dim <= input_rank,
        "Invalid dimension for stack operation. dim=" + std::to_string(dim) + ", limits=[0, " +
            std::to_string(input_rank) + "]");

    for (const auto &input : inputs)
    {
        auto reshaped = dc.op(graphlib::OpType("unsqueeze", {{"dim", dim}}), {input});
        unsqueezed_inputs.push_back(reshaped);
    }

    // Concatenate along the stack dimension
    auto result = dc.op(graphlib::OpType("concatenate", {{"dim", dim}}), unsqueezed_inputs);
    dc.fuse(result);
}

}  // namespace stack
}  // namespace ops
}  // namespace tt
