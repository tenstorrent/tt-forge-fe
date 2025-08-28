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
namespace squeeze
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Squeeze, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Squeeze should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 1, "Squeeze should have one attr.");

    int dim = op.attr_as<int>("dim");

    return torch::squeeze(tensors[0], dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Squeeze, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Squeeze should have single input shape");
    TT_ASSERT(op.attrs().size() == 1, "Squeeze should have one attr.");

    int dim = op.attr_as<int>("dim");

    std::vector<std::uint32_t> output_shape = in_shapes[0];

    // Handle negative dimension
    if (dim < 0)
    {
        dim += output_shape.size();
    }

    TT_ASSERT(dim >= 0 && dim < (int)output_shape.size(), "Dimension index out of bounds");
    TT_ASSERT(output_shape[dim] == 1, "Can only squeeze dimensions of size 1");

    // Remove the dimension
    output_shape.erase(output_shape.begin() + dim);

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
    TT_ASSERT(inputs.size() == 1, "Squeeze should have single input");
    TT_ASSERT(operand == 0, "Squeeze has only one operand");
    TT_ASSERT(op.attrs().size() == 1, "Squeeze should have one attr.");

    // Create unsqueeze operation to restore the squeezed dimension
    int dim = op.attr_as<int>("dim");
    return ac.autograd->create_op(ac, Op("unsqueeze", {{"dim", dim}}), {gradient});
}

}  // namespace squeeze
}  // namespace ops
}  // namespace tt
