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
namespace repeat
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Repeat, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Repeat should have one input tensor.");

    std::vector<int> repeats_vec = op.attr_as<std::vector<int>>("repeats");
    std::vector<int64_t> repeats(repeats_vec.begin(), repeats_vec.end());

    return tensors[0].repeat(repeats);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Repeat, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Repeat should have one input shape.");

    std::vector<int> repeats = op.attr_as<std::vector<int>>("repeats");

    const auto &input_shape = in_shapes[0];

    TT_ASSERT(
        repeats.size() >= input_shape.size(),
        "Repeats vector must have more or equal elements than input shape dimensions");

    // Pad input shape with 1s if needed
    std::vector<std::uint32_t> adjusted_input_shape(repeats.size() - input_shape.size(), 1);
    adjusted_input_shape.insert(adjusted_input_shape.end(), input_shape.begin(), input_shape.end());

    // Calculate output shape
    std::vector<std::uint32_t> output_shape;
    output_shape.reserve(repeats.size());
    for (size_t i = 0; i < repeats.size(); i++) output_shape.push_back(adjusted_input_shape[i] * repeats[i]);

    return {Shape::create(output_shape), {}};
}

// TODO: Implement backward pass
NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Repeat, "Wrong op type.");
    TT_THROW("Repeat backward is not implemented");
    unreachable();
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Repeat, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Repeat should have one input.");

    auto repeats = op.attr_as<std::vector<int>>("repeats");
    auto input_shape = inputs[0].shape.as_vector();

    TT_ASSERT(
        repeats.size() >= input_shape.size(),
        "Repeats vector must have more or equal elements than input shape dimensions");
    auto result = inputs[0];

    // If input has fewer dimensions than repeats, reshape to match dimensions
    if (input_shape.size() < repeats.size())
    {
        std::vector<std::uint32_t> new_shape(repeats.size() - input_shape.size(), 1);
        new_shape.insert(new_shape.end(), input_shape.begin(), input_shape.end());

        result = dc.op(Op("reshape", {{"shape", std::vector<int>(new_shape.begin(), new_shape.end())}}), {result});
        result = dc.op(Op("repeat", {{"repeats", repeats}}), {result});
        dc.fuse(result);
    }
}

}  // namespace repeat
}  // namespace ops
}  // namespace tt
