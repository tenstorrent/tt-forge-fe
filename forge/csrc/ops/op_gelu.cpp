// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

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
namespace gelu
{

tt::graphlib::NodeContext gelu_backward_none(
    tt::autograd::autograd_context &ac,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &gradient)
{
    // d/dx gelu(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
    auto half = ac.autograd->create_constant(ac, 0.5);
    auto one = ac.autograd->create_constant(ac, 1.0);
    auto sqrt1_2 = ac.autograd->create_constant(ac, M_SQRT1_2);
    auto sqrt2pi_factor = ac.autograd->create_constant(ac, 0.5 * M_SQRT1_2 * M_2_SQRTPI);
    auto neg_half = ac.autograd->create_constant(ac, -0.5);

    // x * sqrt(1/2)
    auto x_scaled = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], sqrt1_2});

    // erf(x * sqrt(1/2))
    auto erf_result = ac.autograd->create_op(ac, Op(OpType::Erf), {x_scaled});

    // 1 + erf(x * sqrt(1/2))
    auto one_plus_erf = ac.autograd->create_op(ac, Op(OpType::Add), {one, erf_result});

    // 0.5 * (1 + erf(x * sqrt(1/2)))
    auto cdf = ac.autograd->create_op(ac, Op(OpType::Multiply), {half, one_plus_erf});

    // x^2
    auto x_squared = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], inputs[0]});

    // -0.5 * x^2
    auto neg_half_x_squared = ac.autograd->create_op(ac, Op(OpType::Multiply), {neg_half, x_squared});

    // exp(-0.5 * x^2)
    auto exp_result = ac.autograd->create_op(ac, Op(OpType::Exp), {neg_half_x_squared});

    // x * exp(-0.5 * x^2) * (0.5 * sqrt(1/2) * 2/sqrt(pi))
    auto x_exp = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], exp_result});
    auto pdf = ac.autograd->create_op(ac, Op(OpType::Multiply), {x_exp, sqrt2pi_factor});

    // cdf + pdf
    auto derivative = ac.autograd->create_op(ac, Op(OpType::Add), {cdf, pdf});

    // derivative * gradient
    return ac.autograd->create_op(ac, Op(OpType::Multiply), {derivative, gradient});
}

tt::graphlib::NodeContext gelu_backward_tanh(
    tt::autograd::autograd_context &ac,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &gradient)
{
    // d/dx gelu_tanh(x) = intermediate_0 + intermediate_1
    // intermediate_0 = 0.5 * (1 + tanh((M_2_SQRTPI / M_SQRT2) * (x + 0.044715 * x^3)))
    // intermediate_1 = x * exp(-0.5 * x * x) * (0.5 * M_2_SQRTPI / M_SQRT2)
    auto half = ac.autograd->create_constant(ac, 0.5);
    auto one = ac.autograd->create_constant(ac, 1.0);
    auto neg_half = ac.autograd->create_constant(ac, -0.5);
    auto tanh_factor = ac.autograd->create_constant(ac, M_2_SQRTPI / M_SQRT2);
    auto cubic_factor = ac.autograd->create_constant(ac, 0.044715);
    auto exp_factor = ac.autograd->create_constant(ac, 0.5 * M_2_SQRTPI / M_SQRT2);

    // === Calculate intermediate_0: 0.5 * (1 + tanh((M_2_SQRTPI / M_SQRT2) * (x + 0.044715 * x^3))) ===

    // x^3
    auto x_squared = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], inputs[0]});
    auto x_cubed = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], x_squared});

    // 0.044715 * x^3
    auto cubic_term = ac.autograd->create_op(ac, Op(OpType::Multiply), {cubic_factor, x_cubed});

    // x + 0.044715 * x^3
    auto inner = ac.autograd->create_op(ac, Op(OpType::Add), {inputs[0], cubic_term});

    // (M_2_SQRTPI / M_SQRT2) * (x + 0.044715 * x^3)
    auto tanh_input = ac.autograd->create_op(ac, Op(OpType::Multiply), {tanh_factor, inner});

    // tanh((M_2_SQRTPI / M_SQRT2) * (x + 0.044715 * x^3))
    auto tanh_result = ac.autograd->create_op(ac, Op(OpType::Tanh), {tanh_input});

    // 1 + tanh(...)
    auto one_plus_tanh = ac.autograd->create_op(ac, Op(OpType::Add), {one, tanh_result});

    // intermediate_0 = 0.5 * (1 + tanh(...))
    auto intermediate_0 = ac.autograd->create_op(ac, Op(OpType::Multiply), {half, one_plus_tanh});

    // === Calculate intermediate_1: x * exp(-0.5 * x * x) * (0.5 * M_2_SQRTPI / M_SQRT2) ===

    // x * x
    auto x_times_x = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], inputs[0]});

    // -0.5 * x * x
    auto neg_half_x_squared = ac.autograd->create_op(ac, Op(OpType::Multiply), {neg_half, x_times_x});

    // exp(-0.5 * x * x)
    auto exp_result = ac.autograd->create_op(ac, Op(OpType::Exp), {neg_half_x_squared});

    // x * exp(-0.5 * x * x)
    auto x_exp = ac.autograd->create_op(ac, Op(OpType::Multiply), {inputs[0], exp_result});

    // intermediate_1 = x * exp(-0.5 * x * x) * (0.5 * M_2_SQRTPI / M_SQRT2)
    auto intermediate_1 = ac.autograd->create_op(ac, Op(OpType::Multiply), {x_exp, exp_factor});

    auto derivative = ac.autograd->create_op(ac, Op(OpType::Add), {intermediate_0, intermediate_1});

    // derivative * gradient
    return ac.autograd->create_op(ac, Op(OpType::Multiply), {derivative, gradient});
}

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Gelu, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Gelu should have one input");
    TT_ASSERT(op.attrs().size() == 1, "Gelu should have one attribute");

    std::string approximate = op.attr_as<std::string>("approximate");

    return torch::gelu(tensors[0], approximate);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Gelu, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Gelu should have one input");

    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>{});
}

// Reference implementation is at pytorch/aten/src/ATen/native/cpu/Activation.cpp
// https://github.com/pytorch/pytorch/blob/4f8b986e28736b59bc46cd0873a0f36fdaa6f5b8/aten/src/ATen/native/cpu/Activation.cpp
tt::graphlib::NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Gelu, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Gelu should have one input");
    TT_ASSERT(operand == 0, "Invalid operand index");
    TT_ASSERT(op.attrs().size() == 1, "Gelu should have one attribute");

    std::string approximate = op.attr_as<std::string>("approximate");

    TT_ASSERT(
        approximate == "none" || approximate == "tanh",
        "Gelu backward only supports 'none' or 'tanh' approximation modes, got: {}",
        approximate);

    if (approximate == "none")
    {
        return gelu_backward_none(ac, inputs, gradient);
    }
    else
    {
        return gelu_backward_tanh(ac, inputs, gradient);
    }
}

}  // namespace gelu
}  // namespace ops
}  // namespace tt
