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
namespace transpose
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Transpose, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Transpose should have single input tensor.");

    int dim0 = op.attr_as<int>("dim0");
    int dim1 = op.attr_as<int>("dim1");

    return torch::transpose(tensors[0], dim0, dim1);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Transpose, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Transpose should have single input shape.");

    int dim0 = op.attr_as<int>("dim0");
    int dim1 = op.attr_as<int>("dim1");

    std::vector<std::uint32_t> shape = in_shapes[0];
    int ndim = static_cast<int>(shape.size());

    // We need to handle negative dimension indices.
    if (dim0 < 0)
        dim0 += ndim;
    if (dim1 < 0)
        dim1 += ndim;

    TT_DBG_ASSERT(dim0 >= 0 && dim0 < ndim, "dim0 out of range");
    TT_ASSERT(dim1 >= 0 && dim1 < ndim, "dim1 out of range");

    std::swap(shape[dim0], shape[dim1]);

    return std::make_tuple(graphlib::Shape::create(shape), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Transpose, "Wrong op type.");
    TT_ASSERT(operand == 0, "Invalid operand index");

    int dim0 = op.attr_as<int>("dim0");
    int dim1 = op.attr_as<int>("dim1");

    // Transpose is its own inverse - apply the same transpose to the gradient
    graphlib::OpType transpose_op("transpose");
    transpose_op.set_attr("dim0", dim0);
    transpose_op.set_attr("dim1", dim1);

    return ac.autograd->create_op(ac, transpose_op, {gradient});
}

}  // namespace transpose
}  // namespace ops
}  // namespace tt
