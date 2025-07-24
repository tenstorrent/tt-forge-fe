// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "autograd/autograd.hpp"

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
    const std::vector<std::vector<std::uint32_t>> &in_shapes);

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

}  // namespace op_common
}  // namespace ops
}  // namespace tt
