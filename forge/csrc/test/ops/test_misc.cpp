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
 * Create Conv2d operation with specified attributes
 */
tt::ops::Op create_conv2d_op(
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups,
    const std::vector<int>& padding,
    bool channel_last)
{
    return tt::ops::Op(
        tt::ops::OpType::Conv2d,
        {{"stride", stride},
         {"dilation", dilation},
         {"groups", groups},
         {"padding", padding},
         {"channel_last", channel_last}});
}

std::vector<OpTestParam> generate_conv2d_test_params()
{
    std::vector<OpTestParam> params;

    // Test Case 1: Basic NCHW convolution, no bias
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {0, 0, 0, 0}, false),
        std::vector<graphlib::Shape>{{1, 3, 8, 8}, {16, 3, 3, 3}});

    // Test Case 2: Basic NCHW convolution, with bias
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {1, 1, 1, 1}, false),
        std::vector<graphlib::Shape>{{1, 3, 8, 8}, {16, 3, 3, 3}, {16}});

    // Test Case 3: NHWC convolution, no bias
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {0, 0, 0, 0}, true),
        std::vector<graphlib::Shape>{{1, 8, 8, 3}, {16, 3, 3, 3}});

    // Test Case 4: NHWC convolution, with bias
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {1, 1, 1, 1}, true),
        std::vector<graphlib::Shape>{{1, 8, 8, 3}, {16, 3, 3, 3}, {16}});

    // Test Case 5: Strided convolution
    params.emplace_back(
        create_conv2d_op({2, 2}, {1, 1}, 1, {0, 0, 0, 0}, false),
        std::vector<graphlib::Shape>{{1, 8, 16, 16}, {32, 8, 3, 3}});

    // Test Case 6: Dilated convolution
    params.emplace_back(
        create_conv2d_op({1, 1}, {2, 2}, 1, {2, 2, 2, 2}, false),
        std::vector<graphlib::Shape>{{1, 4, 12, 12}, {8, 4, 3, 3}});

    // Test Case 7: 1x1 convolution
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {0, 0, 0, 0}, false),
        std::vector<graphlib::Shape>{{1, 32, 8, 8}, {64, 32, 1, 1}});

    // Test Case 8: Groups convolution (groups=2)
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 2, {1, 1, 1, 1}, false),
        std::vector<graphlib::Shape>{{1, 8, 6, 6}, {16, 4, 3, 3}});

    // Test Case 9: Asymmetric padding
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {1, 2, 1, 2}, false),
        std::vector<graphlib::Shape>{{1, 2, 4, 4}, {4, 2, 3, 3}});

    // Test Case 10: Larger batch size
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {1, 1, 1, 1}, false),
        std::vector<graphlib::Shape>{{4, 16, 8, 8}, {32, 16, 3, 3}, {32}});

    return params;
}

std::vector<OpTestParam> generate_conv2d_edge_params()
{
    std::vector<OpTestParam> params;

    // Edge Case 1: Minimal input size
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {0, 0, 0, 0}, false),
        std::vector<graphlib::Shape>{{1, 1, 3, 3}, {1, 1, 3, 3}});

    // Edge Case 2: Large kernel
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 1, {2, 2, 2, 2}, false),
        std::vector<graphlib::Shape>{{1, 8, 8, 8}, {16, 8, 5, 5}});

    // Edge Case 3: High stride
    params.emplace_back(
        create_conv2d_op({3, 3}, {1, 1}, 1, {0, 0, 0, 0}, false),
        std::vector<graphlib::Shape>{{1, 4, 12, 12}, {8, 4, 3, 3}});

    // Edge Case 4: Mixed stride/dilation
    params.emplace_back(
        create_conv2d_op({2, 1}, {1, 2}, 1, {1, 1, 1, 1}, false),
        std::vector<graphlib::Shape>{{1, 3, 10, 10}, {6, 3, 3, 3}});

    // Edge Case 5: Groups with bias
    params.emplace_back(
        create_conv2d_op({1, 1}, {1, 1}, 4, {1, 1, 1, 1}, false),
        std::vector<graphlib::Shape>{{1, 12, 6, 6}, {24, 3, 3, 3}, {24}});

    return params;
}

// Conv2d doesn't support backward in current implementation

// Individual test suite - focused representative cases
INSTANTIATE_TEST_SUITE_P(
    Conv2dOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_conv2d_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Edge cases test suite - focused edge scenarios
INSTANTIATE_TEST_SUITE_P(
    Conv2dOpsEdgeCases,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_conv2d_edge_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

/**
 * Create Conv2dTranspose operation with specified attributes
 */
tt::ops::Op create_conv2d_transpose_op(
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups,
    const std::vector<int>& padding,
    const std::vector<int>& output_padding,
    bool channel_last)
{
    return tt::ops::Op(
        tt::ops::OpType::Conv2dTranspose,
        {{"stride", stride},
         {"dilation", dilation},
         {"groups", groups},
         {"padding", padding},
         {"output_padding", output_padding},
         {"channel_last", channel_last}});
}

std::vector<OpTestParam> generate_conv2d_transpose_test_params()
{
    std::vector<OpTestParam> params;

    // Test Case 1: Basic NCHW transpose convolution, no bias
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 1, {0, 0}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 3, 6, 6}, {3, 16, 3, 3}});

    // Test Case 2: NHWC transpose convolution, no bias
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 1, {0, 0}, {0, 0}, true),
        std::vector<graphlib::Shape>{{1, 6, 6, 3}, {3, 16, 3, 3}});

    // Test Case 3: Strided transpose convolution (upsampling)
    params.emplace_back(
        create_conv2d_transpose_op({2, 2}, {1, 1}, 1, {0, 0}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 8, 4, 4}, {8, 32, 3, 3}});

    // Test Case 4: 1x1 transpose convolution
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 1, {0, 0}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 16, 8, 8}, {16, 32, 1, 1}});

    // Test Case 5: With padding transpose
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 1, {1, 1}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 4, 6, 6}, {4, 8, 3, 3}});

    // Test Case 6: With bias tensor (tests eval bias handling + decompose bias logic)
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 1, {0, 0}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 4, 5, 5}, {4, 8, 3, 3}, {8}});

    // Test Case 7: NHWC with bias (tests decompose bias channel alignment)
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 1, {0, 0}, {0, 0}, true),
        std::vector<graphlib::Shape>{{1, 5, 5, 4}, {4, 6, 3, 3}, {6}});

    return params;
}

std::vector<OpTestParam> generate_conv2d_transpose_edge_params()
{
    std::vector<OpTestParam> params;

    // Edge Case 1: Minimal input size
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 1, {0, 0}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 1, 3, 3}, {1, 2, 3, 3}});

    // Edge Case 2: Transpose with output padding
    params.emplace_back(
        create_conv2d_transpose_op({2, 2}, {1, 1}, 1, {0, 0}, {1, 1}, false),
        std::vector<graphlib::Shape>{{1, 4, 3, 3}, {4, 8, 3, 3}});

    // Edge Case 3: Groups convolution (groups=2)
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {1, 1}, 2, {0, 0}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 6, 4, 4}, {6, 8, 3, 3}});

    // Edge Case 4: Dilated transpose convolution
    params.emplace_back(
        create_conv2d_transpose_op({1, 1}, {2, 2}, 1, {0, 0}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 3, 4, 4}, {3, 8, 3, 3}});

    // Edge Case 5: Asymmetric stride and padding
    params.emplace_back(
        create_conv2d_transpose_op({2, 1}, {1, 1}, 1, {1, 2}, {0, 0}, false),
        std::vector<graphlib::Shape>{{1, 4, 5, 8}, {4, 6, 3, 3}});

    // Edge Case 6: Complex combination (stride + padding + output_padding)
    params.emplace_back(
        create_conv2d_transpose_op({2, 2}, {1, 1}, 1, {1, 1}, {1, 0}, false),
        std::vector<graphlib::Shape>{{1, 3, 4, 4}, {3, 8, 3, 3}});

    return params;
}

// Conv2dTranspose doesn't support backward in current implementation

// Individual test suite - focused representative cases
INSTANTIATE_TEST_SUITE_P(
    Conv2dTransposeOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_conv2d_transpose_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Edge cases test suite - focused edge scenarios
INSTANTIATE_TEST_SUITE_P(
    Conv2dTransposeOpsEdgeCases,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_conv2d_transpose_edge_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::misc
