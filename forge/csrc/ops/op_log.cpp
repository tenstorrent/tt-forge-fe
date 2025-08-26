// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace log
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Log, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Log should have one input");

    // Add small epsilon to avoid log(0)
    at::Tensor input_with_epsilon = tensors[0] + 1e-10;
    return torch::log(input_with_epsilon);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Log, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Log should have one input");
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Log, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Log should have one input");
    TT_ASSERT(operand == 0, "Invalid operand index");

    // dx = 1 / x * grad
    auto recip = ac.autograd->create_op(ac, Op("reciprocal"), {inputs[0]});
    return ac.autograd->create_op(ac, Op("multiply"), {recip, gradient});
}

}  // namespace log
}  // namespace ops
}  // namespace tt
