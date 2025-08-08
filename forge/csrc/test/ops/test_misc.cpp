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

}  // namespace tt::test::ops::misc
