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
namespace cumulative_sum
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::CumulativeSum, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Cumulative sum should have one input.");
    TT_ASSERT(op.attrs().size() == 1, "Cumulative sum should have one attribute.");

    return torch::cumsum(tensors[0], op.attr_as<int>("dim"));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::CumulativeSum, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Cumulative sum should have one input.");
    TT_ASSERT(op.attrs().size() == 1, "Cumulative sum should have one attribute.");

    return {Shape::create(in_shapes[0]), {}};
}

/**
 * The backward pass of cumsum is the "reverse cumsum" or cumsum in reverse order.
 * For cumsum along dim, the gradient flows backwards:
 * If y = cumsum(x, dim), then dy/dx[i] = sum(grad_output[j] for j >= i along dim)
 */
NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::CumulativeSum, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Cumulative sum should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for cumsum.");
    TT_ASSERT(op.has_attr("dim"), "Cumulative sum should have one attribute.");

    /**
     * We can not implement backward since we don't have flip operation.
     * The proper mathematical backward pass would require:
     * flip(cumsum(flip(grad, dim), dim), dim) - which computes reverse cumsum.
     *
     * Here is a full implementation:
     *
     * int dim = op.attr_as<int>("dim");

     * // Step 1: Flip the gradient along the dimension
     * auto flipped_grad = ac.autograd->create_op(ac, graphlib::OpType("flip", {}, {{"dims", std::vector<int>{dim}}}),
     * {gradient});
     *
     * // Step 2: Apply cumsum to the flipped gradient
     * auto cumsum_flipped = ac.autograd->create_op(ac, graphlib::OpType("cumsum", {}, {{"dim", dim}}), {flipped_grad});
     *
     * // Step 3: Flip the result back to get the final gradient
     * return ac.autograd->create_op(ac, graphlib::OpType("flip", {}, {{"dims", std::vector<int>{dim}}}),
     * {cumsum_flipped});
     */

    TT_ASSERT(false, "Cumsum does not have backward implemented.");
    unreachable();
}

}  // namespace cumulative_sum
}  // namespace ops
}  // namespace tt
