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
namespace embedding_bw
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::EmbeddingBw, "Wrong op type.");
    TT_ASSERT(tensors.size() == 3, "EmbeddingBw should have exactly 3 input tensors.");

    at::Tensor input = tensors[0];
    at::Tensor weight = tensors[1];
    at::Tensor grad = tensors[2];

    at::Tensor result = torch::zeros(weight.sizes(), weight.options());
    for (int64_t i = 0; i < input.numel(); ++i)
    {
        int64_t idx = input.flatten()[i].item<int64_t>();
        result[idx] = grad.flatten()[i];
    }
    return result;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::EmbeddingBw, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 3, "EmbeddingBw should have exactly 3 input shapes.");

    return std::make_tuple(Shape::create(in_shapes[1]), std::vector<DimBroadcast>{});
}

NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::EmbeddingBw, "Wrong op type.");
    TT_ASSERT(false, "embedding_bw should not be backwarded");
    return gradient;
}

}  // namespace embedding_bw
}  // namespace ops
}  // namespace tt
