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
namespace pixel_shuffle
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::PixelShuffle, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Pixel shuffle should have one input.");

    return torch::nn::functional::pixel_shuffle(tensors[0], op.attr_as<int>("upscale_factor"));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::PixelShuffle, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Pixel shuffle should have one input.");

    auto input_shape = in_shapes[0];
    auto upscale_factor = op.attr_as<int>("upscale_factor");

    TT_ASSERT(input_shape.size() >= 3, "Pixel shuffle should be at least 3D.");

    // Check that channel dimension is divisible by upscale_factor^2
    uint32_t channel_dim = input_shape[input_shape.size() - 3];  // -3rd dimension
    TT_ASSERT(
        channel_dim % (upscale_factor * upscale_factor) == 0,
        "Channel dimension should be divisible by upscale_factor^2");

    std::vector<uint32_t> output_shape(input_shape.begin(), input_shape.end() - 3);

    // Transform: (N, C*r*r, H, W) -> (N, C, H*r, W*r)
    output_shape.push_back(channel_dim / (upscale_factor * upscale_factor));       // C
    output_shape.push_back(input_shape[input_shape.size() - 2] * upscale_factor);  // H * r
    output_shape.push_back(input_shape[input_shape.size() - 1] * upscale_factor);  // W * r

    return {Shape::create(output_shape), {}};
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
    TT_DBG_ASSERT(op.type() == OpType::PixelShuffle, "Wrong op type.");
    TT_THROW("Pixel shuffle does not have backward implemented.");
    unreachable();
}

void decompose_initial(
    const graphlib::OpType &old_op_type, const Op &op, DecomposingContext &dc, const std::vector<NodeContext> &inputs)
{
    TT_DBG_ASSERT(op.type() == OpType::PixelShuffle, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Pixel shuffle should have one input.");

    auto upscale_factor = op.attr_as<int>("upscale_factor");
    auto result = inputs[0];  // Shape: (..., C*r*r, H, W) - any number of leading dims

    auto input_shape = result.shape;
    TT_ASSERT(input_shape.size() >= 3, "Pixel shuffle should have at least 3D input for decompose.");

    // Get last 3 dimensions: (C*r*r, H, W)
    uint32_t C_r2 = input_shape[input_shape.size() - 3];
    uint32_t H = input_shape[input_shape.size() - 2];
    uint32_t W = input_shape[input_shape.size() - 1];
    uint32_t r = upscale_factor;
    uint32_t C = C_r2 / (r * r);

    // Preserve all leading dimensions
    std::vector<int> leading_dims;
    for (size_t i = 0; i < input_shape.size() - 3; ++i) leading_dims.push_back(static_cast<int>(input_shape[i]));

    // Step 1: Reshape to (..., C, r, r, H, W)
    std::vector<int> reshape_dims1 = leading_dims;
    reshape_dims1.push_back(static_cast<int>(C));
    reshape_dims1.push_back(static_cast<int>(r));
    reshape_dims1.push_back(static_cast<int>(r));
    reshape_dims1.push_back(static_cast<int>(H));
    reshape_dims1.push_back(static_cast<int>(W));

    auto x = dc.op(graphlib::OpType("reshape", {{"shape", reshape_dims1}}), {result});

    // Step 2: Transpose sequence - adjust indices based on total dimensions
    int base_dim = static_cast<int>(leading_dims.size());  // offset for leading dimensions

    // First transpose: swap r and H -> [..., C, H, r, r, W]
    x = dc.op(graphlib::OpType("transpose", {{"dim0", base_dim + 1}, {"dim1", base_dim + 3}}), {x});

    // Second transpose: swap r and r -> [..., C, H, r, r, W] (positions changed after first transpose)
    x = dc.op(graphlib::OpType("transpose", {{"dim0", base_dim + 2}, {"dim1", base_dim + 3}}), {x});

    // Third transpose: swap the last two dimensions (r and W) -> [..., C, H, r, W, r]
    x = dc.op(graphlib::OpType("transpose", {{"dim0", base_dim + 3}, {"dim1", base_dim + 4}}), {x});

    // Step 3: Final reshape to (..., C, H * r, W * r)
    std::vector<int> final_dims = leading_dims;
    final_dims.push_back(static_cast<int>(C));
    final_dims.push_back(static_cast<int>(H * r));
    final_dims.push_back(static_cast<int>(W * r));

    x = dc.op(graphlib::OpType("reshape", {{"shape", final_dims}}), {x});

    dc.fuse(x);
}

}  // namespace pixel_shuffle
}  // namespace ops
}  // namespace tt
