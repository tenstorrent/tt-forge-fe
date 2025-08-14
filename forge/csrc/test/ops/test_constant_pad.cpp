// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::constant_pad
{

std::vector<OpTestParam> generate_constant_pad_test_params()
{
    std::vector<OpTestParam> params;

    // Test shapes for constant_pad operation
    auto input_shapes = {
        graphlib::Shape{1, 3, 8},       // 3D: basic case
        graphlib::Shape{2, 4, 6},       // 3D: different dimensions
        graphlib::Shape{1, 3, 8, 12},   // 4D: typical case
        graphlib::Shape{2, 5, 10, 16},  // 4D: larger case
        graphlib::Shape{1, 1, 1, 1},    // 4D: minimal case
        graphlib::Shape{3, 7, 14, 21},  // 4D: larger asymmetric case
    };

    // Different padding configurations in TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]
    std::vector<std::vector<int>> padding_configs = {
        // 3D tensor padding (only last 2 dimensions can be padded)
        {0, 0, 1, 1, 2, 2},  // Symmetric padding on last 2 dims
        {0, 0, 2, 3, 1, 4},  // Asymmetric padding
        {0, 0, 0, 0, 1, 1},  // Only last dimension
        {0, 0, 2, 2, 0, 0},  // Only second-to-last dimension
        {0, 0, 1, 0, 0, 2},  // Sparse padding

        // 4D tensor padding
        {0, 0, 0, 0, 1, 1, 2, 2},  // Symmetric on last 2 dims
        {0, 0, 0, 0, 2, 3, 1, 4},  // Asymmetric on last 2 dims
        {0, 0, 0, 0, 0, 0, 3, 3},  // Only last dimension
        {0, 0, 0, 0, 2, 2, 0, 0},  // Only second-to-last dimension
        {0, 0, 0, 0, 1, 0, 0, 1},  // Diagonal padding
        {0, 0, 0, 0, 5, 2, 3, 1},  // Large asymmetric padding
    };

    // Different constant values
    std::vector<float> values = {0.0f, 1.0f, -1.0f, 2.5f, -3.7f, 42.0f};

    // Generate test parameters
    for (const auto& shape : input_shapes)
    {
        for (const auto& padding : padding_configs)
        {
            // Skip invalid combinations (padding must match tensor rank * 2)
            if (padding.size() != shape.size() * 2)
                continue;

            for (float value : values)
            {
                tt::ops::Op op(tt::ops::OpType::ConstantPad, {{"padding", padding}, {"value", value}});
                params.emplace_back(op, std::vector{shape});
            }
        }
    }

    return params;
}

// Test zero padding cases (should be optimized to nop-like behavior)
std::vector<OpTestParam> generate_zero_padding_params()
{
    std::vector<OpTestParam> params;

    auto input_shapes = {graphlib::Shape{1, 3, 8}, graphlib::Shape{2, 4, 6, 10}};

    for (const auto& shape : input_shapes)
    {
        // Create zero padding vector for this shape
        std::vector<int> zero_padding(shape.size() * 2, 0);

        for (float value : {0.0f, 1.0f, -1.0f})
        {
            tt::ops::Op op(tt::ops::OpType::ConstantPad, {{"padding", zero_padding}, {"value", value}});
            params.emplace_back(op, std::vector{shape});
        }
    }

    return params;
}

// Test single-dimension padding
std::vector<OpTestParam> generate_single_dim_padding_params()
{
    std::vector<OpTestParam> params;

    graphlib::Shape shape{2, 3, 4, 5};  // 4D shape for testing

    // Test padding on each dimension individually
    std::vector<std::vector<int>> single_dim_configs = {
        {2, 1, 0, 0, 0, 0, 0, 0},  // Only dim 0
        {0, 0, 1, 2, 0, 0, 0, 0},  // Only dim 1
        {0, 0, 0, 0, 3, 1, 0, 0},  // Only dim 2
        {0, 0, 0, 0, 0, 0, 2, 3},  // Only dim 3
    };

    for (const auto& padding : single_dim_configs)
    {
        tt::ops::Op op(tt::ops::OpType::ConstantPad, {{"padding", padding}, {"value", 1.5f}});
        params.emplace_back(op, std::vector{shape});
    }

    return params;
}

// Test maximum padding scenarios
std::vector<OpTestParam> generate_large_padding_params()
{
    std::vector<OpTestParam> params;

    // Small tensor with large padding
    graphlib::Shape small_shape{1, 2, 3};
    std::vector<std::vector<int>> large_padding_configs = {
        {0, 0, 5, 5, 10, 10},  // Large symmetric padding
        {0, 0, 10, 2, 3, 15},  // Large asymmetric padding
    };

    for (const auto& padding : large_padding_configs)
    {
        tt::ops::Op op(tt::ops::OpType::ConstantPad, {{"padding", padding}, {"value", -2.0f}});
        params.emplace_back(op, std::vector{small_shape});
    }

    return params;
}

// Instantiate the main constant_pad test suite (decompose only - no backward support yet)
INSTANTIATE_TEST_SUITE_P(
    ConstantPadOpsMain,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_constant_pad_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Instantiate zero padding tests (decompose only - no backward support yet)
INSTANTIATE_TEST_SUITE_P(
    ConstantPadZeroPadding,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_zero_padding_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Instantiate single dimension tests (decompose only - no backward support yet)
INSTANTIATE_TEST_SUITE_P(
    ConstantPadSingleDim,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_single_dim_padding_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Instantiate large padding tests (decompose only - no backward support yet)
INSTANTIATE_TEST_SUITE_P(
    ConstantPadLargePadding,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_large_padding_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::constant_pad
