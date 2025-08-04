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
namespace tanh
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Tanh, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Tanh should have one input.");

    return torch::tanh(tensors[0]);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Tanh, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Tanh should have one input.");

    return {Shape::create(in_shapes[0]), {}};
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
    TT_DBG_ASSERT(op.type() == OpType::Tanh, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Tanh should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for tanh.");

    // d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
    // We can use the output (which is tanh(x)) to compute this efficiently

    // Compute tanh²(x)
    auto tanh_squared = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {output, output});

    // Compute 1 - tanh²(x)
    auto one = ac.autograd->create_constant(ac, 1.0f);
    auto derivative = ac.autograd->create_op(ac, graphlib::OpType("subtract"), {one, tanh_squared});

    // Apply chain rule: derivative * gradient
    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {derivative, gradient});
}

}  // namespace tanh
}  // namespace ops
}  // namespace tt
