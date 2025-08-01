// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace erf
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Erf, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Erf should have one input.");

    return torch::erf(tensors[0]);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Erf, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Erf should have one input.");

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
    TT_DBG_ASSERT(op.type() == OpType::Erf, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Erf should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for erf.");

    // d/dx erf(x) = (2/√π) * exp(-x²)
    // Create constant: 2/√π ≈ 1.1283791670955126
    auto two_over_sqrt_pi = ac.autograd->create_constant(ac, 1.1283791670955126f);

    // Compute x²
    auto x_squared = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {inputs[0], inputs[0]});

    // Compute -x²
    auto neg_one = ac.autograd->create_constant(ac, -1.0f);
    auto neg_x_squared = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {neg_one, x_squared});

    // Compute exp(-x²)
    auto exp_neg_x_squared = ac.autograd->create_op(ac, graphlib::OpType("exp"), {neg_x_squared});

    // Compute derivative: (2/√π) * exp(-x²)
    auto derivative = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {two_over_sqrt_pi, exp_neg_x_squared});

    // Apply chain rule: derivative * gradient
    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {derivative, gradient});
}

}  // namespace erf
}  // namespace ops
}  // namespace tt
