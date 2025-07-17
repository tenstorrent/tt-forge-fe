// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ops/zeros.h>

#include "autograd/autograd.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace constant
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Constant, "Wrong op type.");
    TT_ASSERT(tensors.size() == 0, "Constant eval should not have any operands");
    TT_ASSERT(op.attrs().size() == 1, "Constant eval should contain 1 attr.");

    return torch::tensor({op.attr_as<float>("c")});
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Constant, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 0, "Constant should not have any operands");

    return std::make_tuple(graphlib::Shape::create({1}), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    tt::autograd::autograd_context &context,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Constant, "Wrong op type.");
    TT_THROW("OpType::Constant does not have backward.");
    unreachable();
}

}  // namespace constant
}  // namespace ops
}  // namespace tt
