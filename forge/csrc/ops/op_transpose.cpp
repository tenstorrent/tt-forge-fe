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
namespace transpose
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Transpose, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Transpose should have single input tensor.");

    // Get dim0 and dim1 from attributes
    int dim0 = op.attr_as<int>("dim0");
    int dim1 = op.attr_as<int>("dim1");

    // Handle negative indices if needed (torch::transpose should handle this automatically)
    return torch::transpose(tensors[0], dim0, dim1);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Transpose, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Transpose should have single input shape.");

    // Get dim0 and dim1 from attributes
    int dim0 = op.attr_as<int>("dim0");
    int dim1 = op.attr_as<int>("dim1");

    // Copy input shape and handle negative indices
    std::vector<std::uint32_t> shape = in_shapes[0];
    int ndim = static_cast<int>(shape.size());

    // Convert negative indices to positive
    if (dim0 < 0)
        dim0 += ndim;
    if (dim1 < 0)
        dim1 += ndim;

    // Validate dimensions
    TT_ASSERT(dim0 >= 0 && dim0 < ndim, "dim0 out of range");
    TT_ASSERT(dim1 >= 0 && dim1 < ndim, "dim1 out of range");

    // Swap dimensions
    std::swap(shape[dim0], shape[dim1]);

    return std::make_tuple(graphlib::Shape::create(shape), std::vector<graphlib::DimBroadcast>{});
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Transpose, "Wrong op type.");
    TT_ASSERT(operand == 0, "Invalid operand index");

    // Get dim0 and dim1 from attributes
    int dim0 = op.attr_as<int>("dim0");
    int dim1 = op.attr_as<int>("dim1");

    // Transpose is its own inverse - apply the same transpose to the gradient
    graphlib::OpType transpose_op("transpose");
    transpose_op.set_attr("dim0", dim0);
    transpose_op.set_attr("dim1", dim1);

    return ac.autograd->create_op(ac, transpose_op, {gradient});
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::Transpose, "Wrong op type.");

    // Transpose is mostly data movement, very low FLOPS
    // Just return 1 per element since it's essentially a permutation
    auto shape_tuple = transpose::shape(op, inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace transpose
}  // namespace ops
}  // namespace tt
