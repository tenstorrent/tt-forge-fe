// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ops/zeros.h>

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
at::Tensor Op::constant_eval(const std::vector<at::Tensor> &tensors) const
{
    // assert len(ops) == 0, "constant should not have any operands"
    // assert len(attr) == 1, "constant should contain single attr repr the const. val"

    // # TODO: add data format
    // const_tensor = torch.zeros([1])
    // const_tensor[0] = attr[0]

    // return const_tensor

    TT_ASSERT(tensors.size() == 0, "Constant eval should not have any operands");
    TT_ASSERT(attrs().size() == 1, "Constant eval should contain 1 attr.");

    auto const_tensor = torch::zeros({1});
    const_tensor[0] = attr_as<float>("c1");

    return const_tensor;
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::constant_shape(
    const std::vector<std::vector<std::uint32_t>> &in_shapes) const
{
    // assert len(ops) == 0, "constant should not have any operands"
    // assert len(attr) == 1, "constant should contain single attr repr the const. val"
    // return [1], []

    TT_ASSERT(in_shapes.size() == 0, "Constant should not have any operands");

    return std::make_tuple(graphlib::Shape::create({1}), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext Op::constant_backward(
    tt::autograd::autograd_context &context,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient) const
{
    TT_THROW("OpType::Constant does not have backward.");
    __builtin_unreachable();
}

}  // namespace ops
}  // namespace tt
