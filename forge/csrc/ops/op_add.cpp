// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "op_interface.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{

namespace ops
{

namespace add
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 2, "OpAdd::eval should have two input tensors.");
    return torch::add(tensors[0], tensors[1]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Add, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "add::shape should have two input shapes.");
    TT_ASSERT(op.attrs().size() == 0, "add::shape should not have any attrs.");

    return op_common::compute_elementwise_binary_shape(in_shapes);
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
    TT_ASSERT(inputs.size() == 2, "Add should have two inputs");
    TT_ASSERT(operand >= 0 && operand < 2, "Invalid operand index for add");

    auto input_shape = inputs[operand].shape;
    auto grad_shape = gradient.shape;

    if (input_shape == grad_shape)
    {
        // For addition, gradient flows through unchanged (after handling broadcasting)
        return ac.autograd->create_op(ac, graphlib::OpType("nop"), {gradient});
    }

    // Shapes don't match, we need to reduce along broadcast dimensions using reduce_sum
    return op_common::reduce_broadcast_dimensions(ac, gradient, input_shape, grad_shape);
}

}  // namespace add
}  // namespace ops
}  // namespace tt
