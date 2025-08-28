// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "ops/op_common.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace leaky_relu
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::LeakyRelu, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "LeakyRelu should have one input");
    TT_ASSERT(op.attrs().size() == 1, "LeakyRelu should have one attribute");

    float negative_slope = op.attr_as<float>("parameter");

    return torch::nn::functional::leaky_relu(
        tensors[0], torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::LeakyRelu, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "LeakyRelu should have one input");

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
    TT_DBG_ASSERT(op.type() == OpType::LeakyRelu, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "LeakyRelu should have one input");
    TT_ASSERT(operand == 0, "Invalid operand index");
    TT_ASSERT(op.attrs().size() == 1, "LeakyRelu should have one attribute");

    float negative_slope = op.attr_as<float>("parameter");

    auto alpha = ac.autograd->create_constant(ac, negative_slope);
    auto zero = ac.autograd->create_constant(ac, 0.0);
    auto neg_one = ac.autograd->create_constant(ac, -1.0);

    auto relu_dx = ac.autograd->create_op(ac, Op("heaviside"), {output, zero});

    auto l_relu_dx = ac.autograd->create_op(ac, Op("multiply"), {output, neg_one});
    l_relu_dx = ac.autograd->create_op(ac, Op("heaviside"), {l_relu_dx, zero});
    l_relu_dx = ac.autograd->create_op(ac, Op("multiply"), {l_relu_dx, alpha});
    l_relu_dx = ac.autograd->create_op(ac, Op("add"), {relu_dx, l_relu_dx});

    return ac.autograd->create_op(ac, Op("multiply"), {l_relu_dx, gradient});
}

}  // namespace leaky_relu
}  // namespace ops
}  // namespace tt
