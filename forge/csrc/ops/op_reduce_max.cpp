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
namespace reduce_max
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "reduce_max should have single input tensor.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_max should have 2 attrs (dim_arg, keep_dim).");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    bool keep_dim = op.attr_as<bool>("keep_dim");

    return std::get<0>(torch::max(tensors[0], dim, keep_dim));
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "reduce_max should have single input shape.");
    TT_ASSERT(op.attrs().size() == 2, "reduce_max should have 2 attrs (dim_arg, keep_dim).");

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
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_max should have single input.");
    TT_ASSERT(operand == 0, "Invalid operand index.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    if (dim < 0)
        dim += inputs[0].shape.size();

    // reduce_max: dx = grad * mask
    // where mask = 1.0 at positions where x was max (x==y), and 0.0 elsewhere
    NodeContext one = ac.autograd->create_constant(ac, 1.0);
    float threshold = 1.0;

    graphlib::OpType subtract_op("subtract");
    NodeContext mask = ac.autograd->create_op(ac, subtract_op, {inputs[0], output});

    graphlib::OpType add_op("add");
    mask = ac.autograd->create_op(ac, add_op, {mask, one});

    graphlib::OpType relu_op("relu");
    relu_op.set_attr("threshold", threshold);
    mask = ac.autograd->create_op(ac, relu_op, {mask});

    graphlib::OpType multiply_op("multiply");
    return ac.autograd->create_op(ac, multiply_op, {gradient, mask});
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "reduce_max should have single input.");

    std::vector<int> dims = op.attr_as<std::vector<int>>("dim_arg");
    int dim = dims[0];
    if (dim < 0)
        dim += inputs[0].shape.size();

    if (inputs[0].shape[dim] == 1)
    {
        NodeContext result = dc.op(graphlib::OpType("nop"), {inputs[0]});
        dc.fuse(result);
    }
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::ReduceMax, "Wrong op type.");

    auto shape_tuple = reduce_max::shape(op, inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace reduce_max
}  // namespace ops
}  // namespace tt
