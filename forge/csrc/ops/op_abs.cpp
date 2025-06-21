// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "torch/extension.h"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
at::Tensor Op::abs_eval(const std::vector<at::Tensor> &tensors) const
{
    TT_ASSERT(tensors.size() == 1, "OpAbs::eval should have single input tensor.");
    return torch::abs(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::abs_shape(
    const std::vector<std::vector<std::uint32_t>> &in_shapes) const
{
    TT_ASSERT(in_shapes.size() == 1, "OpAbs::shape should have single input shape.");
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext Op::abs_backward(
    tt::autograd::autograd_context ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    tt::graphlib::NodeContext output,
    tt::graphlib::NodeContext gradient) const
{
    // "op",
    // [](tt::autograd::autograd_context &self,
    // std::variant<std::string, py::object> const &type,
    // std::vector<tt::autograd::NodeContext> operands,
    // std::vector<graphlib::OpType::Attr> attributes,
    // ForgeOpAttrs named_attrs = {})

    // assert len(inputs) == 1, "Abs should have one input"
    // assert operand == 0, "Invalid operand index"
    // heaviside = ac.op("heaviside", (inputs[0], ac.constant(0.5)))
    // subtract = ac.op("subtract", (heaviside, ac.constant(0.5)))
    // stretched = ac.op("multiply", (subtract, ac.constant(2.0)))
    // return ac.op("multiply", (stretched, grad))

    TT_ASSERT(inputs.size() == 1, "Abs should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    auto hs = ac.autograd->create_constant(ac, 0.5);
    auto heaviside = ac.autograd->create_op(ac, graphlib::OpType("heaviside"), {inputs[0], hs});

    auto st = ac.autograd->create_constant(ac, 0.5);
    auto subtract = ac.autograd->create_op(ac, graphlib::OpType("subtract"), {heaviside, st});

    auto sch = ac.autograd->create_constant(ac, 2.0);
    auto stretched = ac.autograd->create_op(ac, graphlib::OpType("multiply"), {subtract, sch});

    return ac.autograd->create_op(ac, graphlib::OpType("multiply"), {stretched, gradient});
}

long Op::abs_initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    auto shape_tuple = shape(inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace ops
}  // namespace tt
