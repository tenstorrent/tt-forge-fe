// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::matmul
{
/**
 * Matmul individual test shapes for testing decompose and backward.
 */
std::vector<VecShapes> get_matmul_individual_test_shapes()
{
    return {
        // 1D tensor tests - commented out because we don't have backward handling.
        // VecShapes{{32}, {32}},         // 1D x 1D: dot product
        // VecShapes{{64}, {64, 32}},     // 1D x 2D: vector-matrix
        // VecShapes{{32, 64}, {64}},     // 2D x 1D: matrix-vector
        // VecShapes{{16}, {2, 16, 32}},  // 1D x 3D: vector-batched matrix
        // VecShapes{{2, 32, 16}, {16}},  // 3D x 1D: batched matrix-vector

        // Basic 2D matmul
        VecShapes{{4, 6}, {6, 8}},
        VecShapes{{8, 12}, {12, 16}},
        VecShapes{{16, 32}, {32, 64}},

        // Batch matmul
        VecShapes{{2, 4, 6}, {2, 6, 8}},
        VecShapes{{3, 8, 12}, {3, 12, 16}},

        // Broadcasting batch dimensions
        VecShapes{{1, 4, 6}, {6, 8}},  // Broadcast batch dim
        VecShapes{{4, 6}, {1, 6, 8}},  // Broadcast batch dim reverse

        // TODO: Complex broadcasting backward pass needs proper support - commented out for now since we don't have
        // backward handling.
        // VecShapes{{1, 4, 6}, {2, 6, 8}},  // Broadcast from 1 to 2
        // VecShapes{{3, 4, 6}, {1, 6, 8}},  // Broadcast from 1 to 3

        // Higher dimensional batch
        VecShapes{{3, 2, 4, 6}, {3, 2, 6, 8}},
        VecShapes{{2, 3, 4, 6}, {2, 3, 6, 8}},
        VecShapes{{1, 1, 4, 6}, {1, 1, 6, 8}},

        // Edge cases
        VecShapes{{1, 1}, {1, 1}},        // Minimal 2D
        VecShapes{{1, 32}, {32, 1}},      // Column/row vectors
        VecShapes{{4, 128}, {128, 256}},  // Larger size
    };
}

/**
 * Sweeps matmul test shapes for testing decompose only.
 */
std::vector<std::vector<graphlib::Shape>> get_matmul_sweeps_test_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    // 1D x 1D combinations
    auto range_1d = shape_range({1}, {8});
    for (const auto& shape : range_1d) input_shapes.push_back({shape, shape});  // Same size 1D vectors for dot product

    // 1D x 2D combinations
    for (const auto& k_dim : std::vector<uint32_t>{2, 4, 8})
        for (const auto& n_dim : std::vector<uint32_t>{2, 4, 8})
            input_shapes.push_back({{k_dim}, {k_dim, n_dim}});  // [K] x [K, N]

    // 2D x 1D combinations
    for (const auto& m_dim : std::vector<uint32_t>{2, 4, 8})
        for (const auto& k_dim : std::vector<uint32_t>{2, 4, 8})
            input_shapes.push_back({{m_dim, k_dim}, {k_dim}});  // [M, K] x [K]

    // 2D x 2D combinations
    auto range_2d_small = shape_range({1, 1}, {4, 4});
    for (const auto& shape1 : range_2d_small)
        for (const auto& shape2 : range_2d_small)
            if (shape1[1] == shape2[0])
                input_shapes.push_back({shape1, shape2});

    // Batched 3D x 3D combinations with small batch sizes
    for (uint32_t batch : {1, 2, 3})
        for (uint32_t m : {2, 4})
            for (uint32_t k : {2, 4})
                for (uint32_t n : {2, 4}) input_shapes.push_back({{batch, m, k}, {batch, k, n}});

    return input_shapes;
}

/**
 * Testing matmul operation.
 */
const std::vector<OpTestParam> get_matmul_test_params(const std::vector<VecShapes>& in_shapes)
{
    std::vector<OpTestParam> params;
    for (const auto& shape_pair : in_shapes) params.emplace_back(tt::ops::Op(tt::ops::OpType::Matmul), shape_pair);

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MatmulOpIndividual,
    SimpleOpTest,
    testing::ValuesIn(get_matmul_test_params(get_matmul_individual_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    MatmulOpSweeps,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_matmul_test_params(get_matmul_sweeps_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::matmul
