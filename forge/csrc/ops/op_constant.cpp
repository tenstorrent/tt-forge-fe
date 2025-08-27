// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ops/zeros.h>

#include "autograd/autograd.hpp"
#include "graph_lib/shape.hpp"
#include "lower_to_forge/common.hpp"
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

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Constant, "Wrong op type.");
    TT_ASSERT(tensors.size() == 0, "Constant eval should not have any operands");
    TT_ASSERT(op.attrs().size() == 1 || op.attrs().size() == 2, "Constant eval should contain 1 or 2 attrs.");
    // Convert from forge dtype to torch dtype
    auto forge_dtype = DataFormat::Float32;
    if (op.has_attr("dtype"))
    {
        forge_dtype = static_cast<DataFormat>(op.attr_as<int>("dtype"));
    }

    auto torch_dtype = graphlib::data_format_to_scalar_type(forge_dtype);

    return torch::tensor({op.attr_as<float>("c")}, torch::TensorOptions().dtype(torch_dtype));
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Constant, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 0, "Constant should not have any operands");

    return std::make_tuple(graphlib::Shape::create({1}), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(

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
