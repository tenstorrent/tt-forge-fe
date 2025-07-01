// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ops/zeros.h>

#include "autograd/autograd.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "torch/extension.h"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
at::Tensor Op::constant_eval(const std::vector<at::Tensor> &tensors) const
{
    TT_DBG_ASSERT(type_ == OpType::Constant, "Wrong op type.");
    TT_DBG_ASSERT(tensors.size() == 0, "Constant eval should not have any operands");
    TT_DBG_ASSERT(attrs().size() == 1, "Constant eval should contain 1 attr.");

    return torch::tensor({attr_as<float>("c")});
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::constant_shape(
    const std::vector<std::vector<std::uint32_t>> &in_shapes) const
{
    TT_DBG_ASSERT(type_ == OpType::Constant, "Wrong op type.");
    TT_DBG_ASSERT(in_shapes.size() == 0, "Constant should not have any operands");

    return std::make_tuple(graphlib::Shape::create({1}), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext Op::constant_backward(
    tt::autograd::autograd_context &context,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient) const
{
    TT_DBG_ASSERT(type_ == OpType::Constant, "Wrong op type.");
    TT_THROW("OpType::Constant does not have backward.");
    __builtin_unreachable();
}

}  // namespace ops
}  // namespace tt
