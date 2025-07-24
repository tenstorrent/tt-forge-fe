// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
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
namespace adv_index
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::AdvIndex, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "AdvIndex should have 2 input tensors.");
    TT_ASSERT(op.attrs().size() == 1, "AdvIndex should have 1 attribute.");

    auto indices = tensors[1];
    TT_ASSERT(indices.dim() == 1 || indices.dim() == 2, "indices should be 1D or 2D");

    int dim = op.attr_as<int>("dim");

    // Convert indices to long tensor and reshape if needed
    at::Tensor indices_long = indices.to(torch::kLong);
    if (indices_long.dim() == 2)
    {
        indices_long = indices_long.reshape(-1);
    }

    return torch::index_select(tensors[0], dim, indices_long);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::AdvIndex, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "AdvIndex should have 2 input shapes.");
    TT_ASSERT(op.attrs().size() == 1, "AdvIndex should have 1 attribute.");
    TT_ASSERT(in_shapes[1].size() == 1 || in_shapes[1].size() == 2, "indices should be 1D or 2D");

    int dim = op.attr_as<int>("dim");
    std::vector<std::uint32_t> data_shape = in_shapes[0];
    std::vector<std::uint32_t> indices_shape = in_shapes[1];

    if (dim < 0)
    {
        dim += data_shape.size();
    }

    auto output_shape = data_shape;
    // Replace the indexed dimension with the size of the indices (last dimension)
    output_shape[dim] = indices_shape.back();

    return std::make_tuple(Shape::create(output_shape), std::vector<DimBroadcast>{});
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
    TT_DBG_ASSERT(op.type() == OpType::AdvIndex, "Wrong op type.");
    TT_ASSERT(operand == 0, "Invalid operand index");

    // adv_index backward is not implemented in Python either
    TT_THROW("Backward pass for adv_index is not implemented");
    unreachable();
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::AdvIndex, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "AdvIndex should have 2 inputs.");
    TT_ASSERT(op.attrs().size() == 1, "AdvIndex should have 1 attribute.");

    int dim = op.attr_as<int>("dim");
    std::vector<uint32_t> data_shape = inputs[0].shape.as_vector<uint32_t>();
    std::vector<int> indices_shape = inputs[1].shape.as_vector<int>();
    TT_ASSERT(indices_shape.size() == 1 || indices_shape.size() == 2, "indices tensor should be 1D or 2D");

    // Idea is to reshape the input tensor to [in0_shape[dim], -1] and then apply the embedding operation
    // The embedding operation will select the appropriate indices from the reshaped tensor
    // and then we will reshape the output back to the original shape.
    //
    // For example:
    // If the input tensor is of shape [N, C, H, W] and we want to index along dim = 2 with indices shape [X],
    // we will first reshape it: [N, C, H, W] -> [N, H, C, W] and [N, H, C, W] -> [H, N, C, W] (permuted)
    // and then reshape it to [H, N * C * W] (flattening the last 3 dimensions)
    // and then apply the embedding operation to select the appropriate indices [H, N * C * W] -> [X, N * C * W].
    // Next, we will reshape the output back to the 4D shape [X, N * C * W] -> [X, N, C, W]
    // and finally, we will transpose the output back to the original order.
    // [X, N, C, W] -> [N, X, C, W] and [N, X, C, W] -> [N, C, X, W]

    // Step 1: Move the indexed dimension to the front using a sequence of transposes
    NodeContext current = inputs[0];
    if (dim != 0)
        for (int i = dim; i > 0; i--)
            current = dc.op(graphlib::OpType("transpose", {}, {{"dim0", i}, {"dim1", i - 1}}), {current});
    NodeContext permuted = current;

    // Step 2: Reshape to [data_shape[dim], -1]
    NodeContext reshaped = permuted;
    if (data_shape.size() != 2)
    {
        // Calculate permuted shape, by popping the element at indexed dim and inserting it at the beginning
        std::vector<std::uint32_t> permuted_shape = data_shape;
        std::uint32_t indexed_dim_shape = permuted_shape[dim];
        permuted_shape.erase(permuted_shape.begin() + dim);
        permuted_shape.insert(permuted_shape.begin(), indexed_dim_shape);

        // Calculate product of all dimensions except the first
        int rest_dims_product = 1;
        for (size_t i = 1; i < permuted_shape.size(); i++)
        {
            rest_dims_product *= permuted_shape[i];
        }

        std::vector<int> reshape_dims = {static_cast<int>(data_shape[dim]), rest_dims_product};
        reshaped = dc.op(graphlib::OpType("reshape", {}, {{"shape", reshape_dims}}), {permuted});
    }

    // Step 3: Apply embedding operation
    // embedding op expects indices tensor as first argument and embedding_table as second argument
    NodeContext selected = dc.op(graphlib::OpType("embedding"), {inputs[1], reshaped});

    // Step 4: Reshape back to appropriate dimensions
    NodeContext reshaped_output = selected;
    if (data_shape.size() != 2)
    {
        std::vector<std::uint32_t> permuted_shape = data_shape;
        std::uint32_t indexed_dim_shape = permuted_shape[dim];
        permuted_shape.erase(permuted_shape.begin() + dim);
        permuted_shape.insert(permuted_shape.begin(), indexed_dim_shape);

        // output_shape = indices_shape (1D) + permuted_shape[1:]
        std::vector<int> output_shape = indices_shape;
        for (size_t i = 1; i < permuted_shape.size(); i++)
        {
            output_shape.push_back(permuted_shape[i]);
        }

        reshaped_output = dc.op(graphlib::OpType("reshape", {}, {{"shape", output_shape}}), {selected});
    }

    // Step 5: Restore original dimension order if necessary using transposes
    NodeContext result = reshaped_output;

    if (dim == 0)
    {
        dc.fuse(result);
        return;
    }

    // Move dimension 0 to position 'dim' using transposes
    current = reshaped_output;
    for (int i = 0; i < dim; i++)
    {
        current = dc.op(graphlib::OpType("transpose", {}, {{"dim0", i}, {"dim1", i + 1}}), {current});
    }
    result = current;
    dc.fuse(result);
}

}  // namespace adv_index
}  // namespace ops
}  // namespace tt
