// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

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
namespace mask
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Mask, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "Mask should have two inputs.");
    TT_ASSERT(op.attrs().size() == 1, "Mask should have one attribute: dim.");

    int dim = op.attr_as<int>("dim");
    at::Tensor inp = tensors[0];
    at::Tensor indices = tensors[1].to(torch::kInt64);

    while (indices.dim() < inp.dim()) indices = indices.unsqueeze(0);

    at::Tensor ones = torch::ones(indices.sizes(), inp.options());
    at::Tensor tensor = torch::zeros(inp.sizes(), inp.options());

    return tensor.scatter_(dim, indices, ones);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Mask, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "Mask should have two inputs.");

    return {Shape::create(in_shapes[0]), {}};
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Mask, "Wrong op type.");
    TT_ASSERT(false, "Mask backward should never be called.");
    unreachable();
}

}  // namespace mask
}  // namespace ops
}  // namespace tt
