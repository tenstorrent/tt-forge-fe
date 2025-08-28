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
namespace embedding
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Embedding, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "Embedding should have exactly 2 input tensors.");

    return torch::embedding(tensors[1], tensors[0].to(torch::kInt32));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Embedding, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "Embedding should have exactly 2 input shapes.");
    TT_ASSERT(in_shapes[1].size() == 2, "Embedding weights should be 2D.");

    // output shape is [*input_shape, embedding_dim]
    std::vector<std::uint32_t> output_shape = in_shapes[0];
    output_shape.push_back(in_shapes[1].back());

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
    TT_DBG_ASSERT(op.type() == OpType::Embedding, "Wrong op type.");

    auto embedding_bw_context = ac.autograd->create_op(ac, Op("embedding_bw"), {inputs[0], inputs[1], gradient});
    embedding_bw_context.output_df = inputs[1].output_df;
    return embedding_bw_context;
}

}  // namespace embedding
}  // namespace ops
}  // namespace tt
