// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ATen/core/TensorBody.h>

#include <cstdint>

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

    // This version takes only the first of multiple maximal values (like pytorch)
    NodeContext one = ac.autograd->create_constant(ac, 1.0);

    // Create negative range tensor: -(torch.arange(in0.shape[dim]) - in0.shape[dim]).float()
    // Example: [3, 2, 1] for dim = 2 and dim_size = 3
    std::uint32_t dim_size = inputs[0].shape[dim];
    at::Tensor neg_range_values = dim_size - torch::arange(dim_size, torch::kFloat32);

    // Create shape for neg_range: [1] * len(in0.shape) with shape[dim] = dim_size
    // Example: [1, 1, 3] for dim = 2 and dim_size = 3
    std::vector<int64_t> shape(inputs[0].shape.size(), 1);
    shape[dim] = dim_size;
    neg_range_values = neg_range_values.reshape(shape);

    NodeContext neg_range = ac.autograd->create_constant(ac, neg_range_values);

    // mask = subtract(in0, output) - has 0.0 in max positions and < 0.0 everywhere else
    graphlib::OpType subtract_op("subtract");
    NodeContext mask = ac.autograd->create_op(ac, subtract_op, {inputs[0], output});

    // mask = add(mask, one) - has 1.0 in max positions and < 1.0 everywhere else
    graphlib::OpType add_op("add");
    mask = ac.autograd->create_op(ac, add_op, {mask, one});

    // mask = greater_equal (mask, one) - has 1.0 in max positions, 0.0 everywhere else
    graphlib::OpType greater_equal_op("greater_equal");
    mask = ac.autograd->create_op(ac, greater_equal_op, {mask, one});

    // mask = multiply(mask, neg_range) - puts range N...1 in max positions, 0.0 everywhere else
    // Example: [1, 1, 0, 1] -> [4, 3, 0, 1]
    graphlib::OpType multiply_op("multiply");
    mask = ac.autograd->create_op(ac, multiply_op, {mask, neg_range});

    // redc = reduce_max(mask) - argmax
    graphlib::OpType reduce_max_op("reduce_max", {}, {{"dim_arg", std::vector<int>{dim}}, {"keep_dim", true}});
    NodeContext redc = ac.autograd->create_op(ac, reduce_max_op, {mask});

    // mask = subtract(mask, redc) - Orig range - argmax, 0.0 in FIRST max position
    mask = ac.autograd->create_op(ac, subtract_op, {mask, redc});

    // mask = add(mask, one) - has 1.0 in first max position, and < 1.0 everywhere else
    mask = ac.autograd->create_op(ac, add_op, {mask, one});

    // mask = greater_equal (mask, one) - has 1.0 in first max position, and 0.0 everywhere else
    mask = ac.autograd->create_op(ac, greater_equal_op, {mask, one});

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
