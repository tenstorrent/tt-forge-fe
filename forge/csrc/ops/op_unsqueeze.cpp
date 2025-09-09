// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
namespace unsqueeze
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Unsqueeze, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Unsqueeze should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 1, "Unsqueeze should have one attr.");

    int dim = op.attr_as<int>("dim");

    return torch::unsqueeze(tensors[0], dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Unsqueeze, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Unsqueeze should have single input shape");
    TT_ASSERT(op.attrs().size() == 1, "Unsqueeze should have one attr.");

    int dim = op.attr_as<int>("dim");

    std::vector<std::uint32_t> output_shape = in_shapes[0];

    // Handle negative dimension
    if (dim < 0)
    {
        dim += output_shape.size() + 1;
    }

    TT_ASSERT(dim >= 0 && dim <= (int)output_shape.size(), "Dimension index out of bounds");
    output_shape.insert(output_shape.begin() + dim, 1);

    return std::make_tuple(graphlib::Shape::create(output_shape), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_ASSERT(inputs.size() == 1, "Unsqueeze should have single input");
    TT_ASSERT(operand == 0, "Unsqueeze has only one operand");

    // Create squeeze operation to reverse the unsqueeze
    int dim = op.attr_as<int>("dim");
    return ac.autograd->create_op(ac, Op(OpType::Squeeze, {{"dim", dim}}), {gradient});
}

}  // namespace unsqueeze
}  // namespace ops
}  // namespace tt
