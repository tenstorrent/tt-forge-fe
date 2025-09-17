// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "lower_to_forge/common.hpp"
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
namespace repeat_interleave
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "RepeatInterleave should have one operand.");

    int repeats = op.attr_as<int>("repeats");
    int dim = op.attr_as<int>("dim");

    if (dim < 0)
    {
        dim += static_cast<int>(tensors[0].dim());
    }
    TT_ASSERT(dim >= 0 && dim < static_cast<int>(tensors[0].dim()), "Given dimension is out of the shape");

    return torch::repeat_interleave(tensors[0], repeats, dim);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "RepeatInterleave should have one operand.");

    int repeats = op.attr_as<int>("repeats");
    int dim = op.attr_as<int>("dim");

    if (dim < 0)
    {
        dim += static_cast<int>(in_shapes[0].size());
    }
    TT_ASSERT(dim >= 0 && dim < static_cast<int>(in_shapes[0].size()), "Given dimension is out of the shape");

    std::vector<std::uint32_t> output_shape = in_shapes[0];
    output_shape[dim] *= repeats;

    return std::make_tuple(Shape::create(output_shape), std::vector<DimBroadcast>{});
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::RepeatInterleave, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "RepeatInterleave should have one operand.");

    int repeats = op.attr_as<int>("repeats");
    int dim = op.attr_as<int>("dim");

    std::vector<int> input_shape = inputs[0].shape.as_vector<int>();
    if (dim < 0)
    {
        dim += static_cast<int>(input_shape.size());
    }
    TT_ASSERT(dim >= 0 && dim < static_cast<int>(input_shape.size()), "Given dimension is out of the shape");

    std::vector<int> shape = input_shape;
    std::vector<int> grad_shape = gradient.shape.as_vector<int>();
    shape[dim] = repeats;
    shape.insert(shape.begin() + dim, grad_shape[dim] / repeats);

    NodeContext reshaped = ac.autograd->create_op(ac, Op(OpType::Reshape, {{"shape", shape}}), {gradient});

    NodeContext reduced = ac.autograd->create_op(
        ac, Op(OpType::ReduceSum, {{"dim_arg", std::vector<int>{dim + 1}}, {"keep_dim", true}}), {reshaped});

    NodeContext squeezed = ac.autograd->create_op(ac, Op(OpType::Squeeze, {{"dim", dim + 1}}), {reduced});

    return squeezed;
}

}  // namespace repeat_interleave
}  // namespace ops
}  // namespace tt
