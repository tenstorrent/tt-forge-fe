// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::misc
{
std::vector<VecShapes> get_cumsum_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 1, 32}},
        VecShapes{{1, 1, 32, 1}},
        VecShapes{{1, 32, 1, 1}},
        VecShapes{{32, 1, 1, 1}},
        VecShapes{{1, 2, 3, 4}},
        VecShapes{{2, 3, 4}},
        VecShapes{{3, 4}},
        VecShapes{{4}},
        VecShapes{{1}}};
}

std::vector<std::vector<graphlib::Shape>> get_cumsum_sweeps_test_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range({1}, {16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {5, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1, 1}, {5, 1, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    for (size_t i = 0; i < shapes.size(); ++i) input_shapes.push_back({shapes[i]});

    return input_shapes;
}

/**
 * Testing cumulative sum operation.
 */
const std::vector<OpTestParam> get_cumsum_test_params(const std::vector<VecShapes>& in_shapes)
{
    std::vector<OpTestParam> params;

    for (const auto& shape_vec : in_shapes)
    {
        for (const auto& shape : shape_vec)
        {
            for (int dim = 0; dim < static_cast<int>(shape.size()); ++dim)
            {
                tt::ops::Op op1(tt::ops::OpType::CumulativeSum, {{"dim", static_cast<int>(dim)}});
                tt::ops::Op op2(tt::ops::OpType::CumulativeSum, {{"dim", -static_cast<int>(dim) - 1}});
                params.emplace_back(op1, std::vector{shape});
                params.emplace_back(op2, std::vector{shape});
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    CumulativeSumOpNoBackwardIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_cumsum_test_params(get_cumsum_individual_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    CumulativeSumOpNoBackwardSweeps,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_cumsum_test_params(get_cumsum_sweeps_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

std::vector<VecShapes> get_mask_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 1, 32}, {1, 1, 1, 32}},
        VecShapes{{1, 1, 32, 1}, {1, 1, 32, 1}},
        VecShapes{{1, 32, 1, 1}, {1, 32, 1, 1}},
        VecShapes{{32, 1, 1, 1}, {32, 1, 1, 1}},
        VecShapes{{1, 2, 3, 4}, {1, 2, 3, 4}},
        VecShapes{{2, 3, 4}, {2, 3, 4}},
        VecShapes{{3, 4}, {3, 4}},
        VecShapes{{4}, {4}},
        VecShapes{{1}, {1}}};
}

std::vector<std::vector<graphlib::Shape>> get_mask_sweeps_test_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range({1}, {16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {5, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1, 1}, {5, 1, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    for (size_t i = 0; i < shapes.size(); ++i) input_shapes.push_back({shapes[i], shapes[i]});

    return input_shapes;
}

/**
 * Testing mask operation.
 */
const std::vector<OpTestParam> get_mask_test_params(const std::vector<VecShapes>& in_shapes)
{
    std::vector<OpTestParam> params;

    for (const auto& shape_vec : in_shapes)
    {
        for (int dim = 0; dim < static_cast<int>(shape_vec[0].size()); ++dim)
        {
            tt::ops::Op op(tt::ops::OpType::Mask, {{"dim", static_cast<int>(dim)}});
            params.emplace_back(op, shape_vec);
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MaskNoBackwardIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_mask_test_params(get_mask_individual_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    MaskOpNoBackwardSweeps,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_mask_test_params(get_mask_sweeps_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

/**
 * Testing update_cache operation.
 */
const std::vector<OpTestParam> get_update_cache_test_params()
{
    std::vector<OpTestParam> params;

    // Test various cache and input configurations
    std::vector<std::tuple<graphlib::Shape, graphlib::Shape, graphlib::Shape>> test_configs = {
        // {cache_shape, input_shape, index_shape}
        {{2, 4, 10, 64}, {2, 4, 1, 64}, {2}},     // Basic case
        {{4, 8, 20, 32}, {2, 8, 1, 32}, {2}},     // Different batch sizes
        {{1, 1, 5, 16}, {1, 1, 1, 16}, {1}},      // Minimal case
        {{8, 12, 50, 128}, {4, 12, 1, 128}, {4}}  // Larger case
    };

    for (const auto& [cache_shape, input_shape, index_shape] : test_configs)
    {
        // Test different batch_offset values
        for (int batch_offset : {0, 1, 2})
        {
            // Only test valid batch_offset values (must allow input to fit in cache)
            if (batch_offset + input_shape[0] <= cache_shape[0])
            {
                tt::ops::Op op(tt::ops::OpType::UpdateCache, {{"batch_offset", batch_offset}});
                params.emplace_back(op, std::vector{cache_shape, input_shape, index_shape});
            }
        }
    }

    return params;
}

/**
 * Testing fill_cache operation.
 */
const std::vector<OpTestParam> get_fill_cache_test_params()
{
    std::vector<OpTestParam> params;

    // Test various cache and input configurations
    std::vector<std::tuple<graphlib::Shape, graphlib::Shape>> test_configs = {
        // {cache_shape, input_shape}
        {{2, 4, 10, 64}, {2, 4, 3, 64}},      // Basic case
        {{4, 8, 20, 32}, {2, 8, 5, 32}},      // Different batch sizes
        {{1, 1, 5, 16}, {1, 1, 2, 16}},       // Minimal case
        {{8, 12, 50, 128}, {4, 12, 10, 128}}  // Larger case
    };

    for (const auto& [cache_shape, input_shape] : test_configs)
    {
        // Test different batch_offset values
        for (int batch_offset : {0, 1, 2})
        {
            // Only test valid batch_offset values (must allow input to fit in cache)
            if (batch_offset + input_shape[0] <= cache_shape[0])
            {
                tt::ops::Op op(tt::ops::OpType::FillCache, {{"batch_offset", batch_offset}});
                params.emplace_back(op, std::vector{cache_shape, input_shape});
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    UpdateCacheOpNoBackward,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_update_cache_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    FillCacheOpNoBackward,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_fill_cache_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

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

}  // namespace tt::test::ops::misc
