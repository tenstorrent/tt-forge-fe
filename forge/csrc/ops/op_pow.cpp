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
namespace pow
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Pow, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Pow should have one input.");
    TT_ASSERT(op.attrs().size() == 1, "Pow should have one attribute: exponent.");

    return torch::pow(tensors[0], op.attr_as<float>("exponent"));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Pow, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Pow should have one input.");

    return {Shape::create(in_shapes[0]), {}};
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Pow, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Pow should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for pow.");
    TT_ASSERT(op.attrs().size() == 1, "Pow should have one attribute: exponent.");

    // d/dx x^n = n * x^(n-1) = n * (x^n / x) = n * output / x
    float exponent = op.attr_as<float>("exponent");

    // Create constant for exponent
    auto exponent_const = ac.autograd->create_constant(ac, exponent);

    // Compute reciprocal of input: 1/x
    auto reciprocal = ac.autograd->create_op(ac, Op(OpType::Reciprocal), {inputs[0]});

    // Compute n * x^n / x = n * output / x
    auto partial_grad = ac.autograd->create_op(ac, Op(OpType::Multiply), {output, reciprocal});
    auto derivative = ac.autograd->create_op(ac, Op(OpType::Multiply), {exponent_const, partial_grad});

    // Apply chain rule: derivative * gradient
    return ac.autograd->create_op(ac, Op(OpType::Multiply), {derivative, gradient});
}

}  // namespace pow
}  // namespace ops
}  // namespace tt
