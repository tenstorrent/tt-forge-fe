// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace tt
{
namespace graphlib
{
class Shape;
using DimBroadcast = std::tuple<int, int, int>;
}  // namespace graphlib

namespace ops
{
namespace common_utils
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

}  // namespace common_utils
}  // namespace ops
}  // namespace tt
