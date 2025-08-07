// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::conv_2d
{

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

/**
 * Generate focused Conv2d test parameters covering key code paths
 */
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

/**
 * Generate additional focused test cases for edge scenarios and dtype testing
 */
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

}  // namespace tt::test::ops::conv_2d
