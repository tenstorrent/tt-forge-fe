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
namespace log_softmax
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::LogSoftmax, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "LogSoftmax should have one operand.");

    int dim = op.attr_as<int>("dim");
    if (dim < 0)
    {
        dim += static_cast<int>(tensors[0].dim());
    }
    TT_ASSERT(dim >= 0 && dim < static_cast<int>(tensors[0].dim()), "Given dimension is out of the shape");

    return torch::log_softmax(tensors[0], dim);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::LogSoftmax, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "LogSoftmax should have one operand.");

    // Check if the dimension is out of the shape
    int dim = op.attr_as<int>("dim");
    if (dim < 0)
    {
        dim += static_cast<int>(in_shapes[0].size());
    }
    TT_ASSERT(dim >= 0 && dim < static_cast<int>(in_shapes[0].size()), "Given dimension is out of the shape");

    return std::make_tuple(Shape::create(in_shapes[0]), std::vector<DimBroadcast>{});
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::LogSoftmax, "Wrong op type.");
    TT_THROW("Back propagation for LogSoftmax op is not implemented yet");
    unreachable();
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::LogSoftmax, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "LogSoftmax should have one operand.");

    NodeContext x = inputs[0];
    int dim = op.attr_as<int>("dim");
    bool stable = op.attr_as<bool>("stable");

    NodeContext softmax_result = dc.op(Op("softmax", {{"dim", dim}, {"stable", stable}}), {x});
    NodeContext result = dc.op(Op("log"), {softmax_result});

    dc.fuse(result);
}

}  // namespace log_softmax
}  // namespace ops
}  // namespace tt
