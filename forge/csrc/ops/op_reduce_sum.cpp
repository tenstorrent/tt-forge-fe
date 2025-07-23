// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace reduce_sum
{

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "reduce_sum should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_sum should have 2 attrs (dim, keep_dim).");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    bool keep_dim = op.attr_as<bool>("keep_dim");

    return torch::sum(tensors[0], dim, keep_dim);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "reduce_sum should have single input shape.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_sum should have 2 attrs (dim, keep_dim).");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    if (dim < 0)
        dim += in_shapes[0].size();

    bool keep_dim = op.attr_as<bool>("keep_dim");
    std::vector<std::uint32_t> ret = in_shapes[0];

    if (keep_dim)
        ret[dim] = 1;
    else
        ret.erase(ret.begin() + dim);

    return std::make_tuple(graphlib::Shape::create(ret), std::vector<graphlib::DimBroadcast>{});
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
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_sum should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    // For sum, gradient just needs to be broadcast back to original shape
    // The broadcast will be implicitly figured out during shape calculations
    graphlib::OpType nop_op("nop");
    return ac.autograd->create_op(ac, nop_op, {gradient});
}

void decompose_initial(
    const graphlib::OpType &old_op_type,
    const Op &op,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceSum, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_sum should have single input.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    if (dim < 0)
        dim += inputs[0].shape.size();

    if (inputs[0].shape.as_vector<int>()[dim] == 1)
    {
        NodeContext result = dc.op(graphlib::OpType("nop"), {inputs[0]});
        dc.fuse(result);
    }
}

}  // namespace reduce_sum
}  // namespace ops
}  // namespace tt
