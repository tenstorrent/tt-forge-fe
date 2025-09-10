// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/torch.h"

namespace tt
{
namespace graphlib
{
class Shape;
using DimBroadcast = std::tuple<int, int, int>;
}  // namespace graphlib

namespace ops
{
namespace op_common
{

/**
 * Compute the output shape and broadcast information for elementwise binary operations.
 * This function implements standard broadcasting rules where dimensions are aligned by appending 1s to the beginning of
 * the shorter shape, and dimensions of size 1 can be broadcasted to match larger dimensions.
 *
 * @param in_shapes Vector containing exactly 2 input shapes
 * @return Tuple containing the output shape and broadcast information
 */
std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> compute_elementwise_binary_shape(
    const std::vector<std::vector<uint32_t>> &in_shapes);

/**
 * Compute the output shape and broadcast information for elementwise nnary operations.
 * This function implements standard broadcasting rules where dimensions are aligned by appending 1s to the beginning of
 * the shorter shape, and dimensions of size 1 can be broadcasted to match larger dimensions.
 *
 * @param in_shapes Vector containing input shapes.
 * @return Tuple containing the output shape and broadcast information.
 */
std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> eltwise_nary_shape(
    const std::vector<std::vector<uint32_t>> &in_shapes);

/**
 * Handle broadcast reduction in backward pass for elementwise binary operations.
 * This function reduces gradients along dimensions where broadcasting occurred during forward pass.
 *
 * @param ac Autograd context for creating operations
 * @param gradient The incoming gradient
 * @param input_shape Shape of the input operand we're computing gradient for
 * @param grad_shape Shape of the incoming gradient
 * @return NodeContext with properly reduced gradient
 */
tt::graphlib::NodeContext reduce_broadcast_dimensions(
    tt::autograd::autograd_context &ac,
    const tt::graphlib::NodeContext &gradient,
    const tt::graphlib::Shape &input_shape,
    const tt::graphlib::Shape &grad_shape);

/**
 * Calculate initial FLOPS estimate for operations based on output shape.
 * This is a common pattern where FLOPS equals the number of output elements.
 *
 * @param shape_tuple Tuple containing the output shape and broadcast information from an operation's shape function
 * @return FLOPS estimate (number of output elements)
 */
long initial_flops_estimate_output_dim(std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape_tuple);

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> reduce_ops_shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes);

/**
 * Convert resize method integer to string.
 *
 * @param method Resize method integer
 * @return Resize method string
 */
std::string get_resize_method(int method);

/**
 * Promotes all floating point tensors to the biggest float type of all tensors.
 * @param tensors input tensors.
 * @return promoted tensors.
 */
std::vector<at::Tensor> promote_floating_dtypes(const std::vector<at::Tensor> &tensors);

struct PaddingParams
{
    int left = 0, right = 0, top = 0, bottom = 0;
    int width_dim, height_dim;

    // Original constructor for standard padding operations
    PaddingParams(const std::vector<int> &padding, int shape_size, bool channel_last)
    {
        width_dim = channel_last ? shape_size - 2 : shape_size - 1;
        height_dim = channel_last ? shape_size - 3 : shape_size - 2;

        if (padding.size() == 2)
        {
            left = padding[0];
            right = padding[1];
        }
        else if (padding.size() == 4)
        {
            left = padding[0];
            right = padding[1];
            top = padding[2];
            bottom = padding[3];
        }
    }

    // Simplified constructor for padding on a specific dimension
    PaddingParams(int dim, int left_pad, int right_pad, int shape_size) :
        left(left_pad), right(right_pad), top(0), bottom(0), width_dim(dim), height_dim(0)
    {
        // Set height_dim to a safe default (could be any dimension not equal to width_dim)
        height_dim = (dim == 0) ? 1 : 0;
        if (height_dim >= shape_size)
        {
            height_dim = (dim + 1) % shape_size;
        }
    }
};

// Helper to create constant tensor based on context type
template <typename ContextType>
tt::graphlib::NodeContext create_constant(ContextType &context, const at::Tensor &tensor)
{
    if constexpr (std::is_same_v<ContextType, tt::DecomposingContext>)
        return tt::DecomposingContext::create_constant_tensor(context, tensor);
    else
        return context.autograd->create_constant_tensor(context, tensor);
}

// Helper to create op based on context type
template <typename ContextType>
tt::graphlib::NodeContext create_op(
    ContextType &context, const ops::Op &op, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    if constexpr (std::is_same_v<ContextType, tt::DecomposingContext>)
        return context.op(op, inputs);
    else
        return context.autograd->create_op(context, op, inputs);
}

// Template function to concatenate patches with optional first and second patches
template <typename ContextType>
tt::graphlib::NodeContext concat_patches(
    ContextType &context,
    const tt::graphlib::NodeContext *first_patch,
    const tt::graphlib::NodeContext &center,
    const tt::graphlib::NodeContext *second_patch,
    int dim_axis)
{
    std::vector<tt::graphlib::NodeContext> inputs;

    if (first_patch)
    {
        inputs.push_back(*first_patch);
    }

    inputs.push_back(center);

    if (second_patch)
    {
        inputs.push_back(*second_patch);
    }

    if (inputs.size() == 1)
    {
        return center;  // No concatenation needed
    }

    return create_op(context, ops::Op(OpType::Concatenate, {{"dim", dim_axis}}), inputs);
}

// Template function for constant mode padding decomposition
template <typename ContextType>
tt::graphlib::NodeContext decompose_constant_mode(
    ContextType &context, const tt::graphlib::NodeContext &input, const PaddingParams &params, float value)
{
    /**
     * This code path fails with memory exceeded error in metal. Example:
     * Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 2197184 B which is beyond max
     * L1 size of 1499136 B.
     *
     * Once fixed, we should enable this code path.
     */
    if constexpr (false)
    {
        // Convert padding format from [left, right] or [left, right, top, bottom]
        // to TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]
        int rank = static_cast<int>(input.shape.size());
        std::vector<int> constant_padding(rank * 2, 0);  // Initialize all to 0

        constant_padding[params.width_dim * 2] = params.left;       // low padding
        constant_padding[params.width_dim * 2 + 1] = params.right;  // high padding

        if (params.top > 0 || params.bottom > 0)
        {
            constant_padding[params.height_dim * 2] = params.top;         // low padding
            constant_padding[params.height_dim * 2 + 1] = params.bottom;  // high padding
        }

        return create_op(
            context, ops::Op(OpType::ConstantPad, {{"padding", constant_padding}, {"value", value}}), {input});
    }

    /**
     * Alternative implementation for constant padding using concatenation.
     * This decomposes constant padding into constant tensor creation, and concatenation operations
     * to avoid the memory issues with direct constant_pad operation.
     */
    tt::graphlib::NodeContext result = input;

    std::unique_ptr<tt::graphlib::NodeContext> left_pad, right_pad, top_pad, bot_pad;

    if (params.left > 0)
    {
        auto left_shape = input.shape.as_vector();
        left_shape[params.width_dim] = params.left;

        at::Tensor left_constant = at::full(
            std::vector<int64_t>(left_shape.begin(), left_shape.end()),
            value,
            at::TensorOptions().dtype(tt::graphlib::data_format_to_scalar_type(input.output_df)));

        left_pad = std::make_unique<tt::graphlib::NodeContext>(create_constant(context, left_constant));
    }

    if (params.right > 0)
    {
        auto right_shape = input.shape.as_vector();
        right_shape[params.width_dim] = params.right;

        at::Tensor right_constant = at::full(
            std::vector<int64_t>(right_shape.begin(), right_shape.end()),
            value,
            at::TensorOptions().dtype(tt::graphlib::data_format_to_scalar_type(input.output_df)));

        right_pad = std::make_unique<tt::graphlib::NodeContext>(create_constant(context, right_constant));
    }

    result = concat_patches(context, left_pad.get(), result, right_pad.get(), params.width_dim);

    if (params.top > 0)
    {
        auto top_shape = result.shape.as_vector();
        top_shape[params.height_dim] = params.top;

        at::Tensor top_constant = at::full(
            std::vector<int64_t>(top_shape.begin(), top_shape.end()),
            value,
            at::TensorOptions().dtype(tt::graphlib::data_format_to_scalar_type(input.output_df)));

        top_pad = std::make_unique<tt::graphlib::NodeContext>(create_constant(context, top_constant));
    }

    if (params.bottom > 0)
    {
        auto bottom_shape = result.shape.as_vector();
        bottom_shape[params.height_dim] = params.bottom;

        at::Tensor bottom_constant = at::full(
            std::vector<int64_t>(bottom_shape.begin(), bottom_shape.end()),
            value,
            at::TensorOptions().dtype(tt::graphlib::data_format_to_scalar_type(input.output_df)));

        bot_pad = std::make_unique<tt::graphlib::NodeContext>(create_constant(context, bottom_constant));
    }

    return concat_patches(context, top_pad.get(), result, bot_pad.get(), params.height_dim);
}
void decompose_nearest_interpolation(
    tt::DecomposingContext &dc, const tt::graphlib::NodeContext &activation, std::vector<int> sizes, bool channel_last);

}  // namespace op_common
}  // namespace ops
}  // namespace tt
