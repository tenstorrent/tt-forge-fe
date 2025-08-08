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
namespace update_cache
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::UpdateCache, "Wrong op type.");
    TT_ASSERT(tensors.size() == 3, "UpdateCache should have three inputs: cache, input, update_index.");
    TT_ASSERT(op.attrs().size() == 1, "UpdateCache should have one attribute: batch_offset.");

    at::Tensor cache = tensors[0];
    at::Tensor input = tensors[1];
    at::Tensor update_index = tensors[2];
    int batch_offset = op.attr_as<int>("batch_offset");

    at::Tensor cache_out = cache.clone();

    // Handle scalar update_index by adding dimension.
    if (update_index.dim() == 0)
        update_index = update_index.unsqueeze(0);

    TT_ASSERT(cache.dim() == 4, "Expected 4D tensor for cache.");
    TT_ASSERT(input.dim() == 4, "Expected 4D tensor for input.");
    TT_ASSERT(update_index.dim() == 1, "Expected 1D tensor for update_index.");

    auto cache_sizes = cache.sizes();
    auto input_sizes = input.sizes();

    int64_t B_cache = cache_sizes[0], H_cache = cache_sizes[1], S_cache = cache_sizes[2], D_cache = cache_sizes[3];
    int64_t B_in = input_sizes[0], H_in = input_sizes[1], S_in = input_sizes[2], D_in = input_sizes[3];

    TT_ASSERT(S_in == 1, "UpdateCache operation requires input sequence length S=1.");
    TT_ASSERT(
        H_cache == H_in && D_cache == D_in, "Number of heads H and hidden dimension D must match for cache and input.");
    TT_ASSERT(update_index.size(0) == B_in, "Update index batch size must match input batch size.");
    TT_ASSERT(batch_offset + B_in <= B_cache, "batch_offset + input batch size exceeds cache batch size.");

    for (int64_t b = 0; b < B_in; ++b)
    {
        int64_t idx = update_index[b].item<int64_t>();
        TT_ASSERT(idx >= 0 && idx < S_cache, "Invalid update index.");

        // Update cache[b + batch_offset, :, idx:idx+1, :] = input[b]
        cache_out.slice(0, b + batch_offset, b + batch_offset + 1)
            .slice(2, idx, idx + 1)
            .copy_(input.slice(0, b, b + 1));
    }

    return cache_out;
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::UpdateCache, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 3, "UpdateCache should have three inputs.");

    return {Shape::create(in_shapes[0]), {}};
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
    TT_DBG_ASSERT(op.type() == OpType::UpdateCache, "Wrong op type.");
    TT_ASSERT(inputs.size() == 3, "UpdateCache should have three inputs.");
    TT_ASSERT(operand == 0, "Invalid operand index for update cache.");

    TT_ASSERT(false, "Update cache does not have backward implemented.");
    unreachable();
}

}  // namespace update_cache
}  // namespace ops
}  // namespace tt
