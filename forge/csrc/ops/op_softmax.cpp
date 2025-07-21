// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

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
namespace softmax
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Softmax, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Softmax should have one operand.");

    int dim = op.attr_as<int>("dim");

    TT_ASSERT(tensors[0].dim() > dim, "Given dimension is out of the shape");

    return torch::softmax(tensors[0], dim);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Softmax, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Softmax should have one operand.");

    return std::make_tuple(Shape::create(in_shapes[0]), std::vector<DimBroadcast>{});
}

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Softmax, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Softmax should have one operand.");

    return ac.autograd->create_op(
        ac, graphlib::OpType("softmax_bw", {}, {{"dim", op.attr_as<int>("dim")}}), {inputs[0], output, gradient});
}

}  // namespace softmax
}  // namespace ops
}  // namespace tt
