// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace softmax
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Softmax, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Softmax should have one operand.");

    int dim = op.attr_as<int>("dim");

    TT_ASSERT(tensors[0].dim() > dim, "Given dimension is out of the shape");

    return torch::softmax(tensors[0], dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Softmax, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Softmax should have one operand.");

    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcastTrampoline>{});
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Softmax, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Softmax should have one operand.");

    int dim = op.attr_as<int>("dim");

    graphlib::OpType softmax_bw_op("softmax_bw");
    softmax_bw_op.set_attr("dim", dim);

    return ac.autograd->create_op(ac, softmax_bw_op, {inputs[0], output, gradient});
}

}  // namespace softmax
}  // namespace ops
}  // namespace tt
