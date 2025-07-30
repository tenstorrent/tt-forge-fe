// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::pooling
{

// Common validation logic for all pooling operations
bool valid_pool_1d_inputs(
    const std::vector<graphlib::Shape>& shapes, int kernel_size, int stride, int padding, int dilation = 1)
{
    if (shapes.size() != 1)
        return false;

    const auto& shape = shapes[0];
    if (shape.size() < 3)
        return false;

    uint32_t l_in = shape[shape.size() - 1];

    // Check if output dimension would be positive
    int l_numerator = static_cast<int>(l_in) + 2 * padding - dilation * (kernel_size - 1) - 1;
    if (l_numerator < 0)
        return false;

    uint32_t l_out = static_cast<uint32_t>(l_numerator / stride + 1);
    return l_out > 0;
}

bool valid_avg_pool_1d_inputs(
    const std::vector<graphlib::Shape>& shapes, int kernel_size, int stride, int padding, int dilation = 1)
{
    // Only support global pooling for now
    if (stride != 1)
        return false;
    if (padding != 0)
        return false;
    if (dilation != 1)
        return false;

    return kernel_size == static_cast<int>(shapes[0][shapes[0].size() - 1]);
}

bool valid_pool_2d_inputs(
    const std::vector<graphlib::Shape>& shapes,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h = 1,
    int dilation_w = 1,
    bool channel_last = false)
{
    if (shapes.size() != 1)
        return false;

    const auto& shape = shapes[0];
    if (shape.size() < 4)
        return false;

    uint32_t h_in = channel_last ? shape[shape.size() - 3] : shape[shape.size() - 2];
    uint32_t w_in = channel_last ? shape[shape.size() - 2] : shape[shape.size() - 1];

    // Check if output dimensions would be positive
    int h_numerator = static_cast<int>(h_in) + 2 * padding_h - dilation_h * (kernel_h - 1) - 1;
    int w_numerator = static_cast<int>(w_in) + 2 * padding_w - dilation_w * (kernel_w - 1) - 1;

    if (h_numerator < 0 || w_numerator < 0)
        return false;

    uint32_t h_out = static_cast<uint32_t>(h_numerator / stride_h + 1);
    uint32_t w_out = static_cast<uint32_t>(w_numerator / stride_w + 1);

    return h_out > 0 && w_out > 0;
}

// Common shape generation for all pooling tests
std::vector<graphlib::Shape> generate_pool_1d_shapes()
{
    std::vector<graphlib::Shape> shapes;

    // 3D shapes: (N, C, L)
    auto shapes_range = shape_range({1, 1, 8}, {2, 8, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    return shapes;
}

std::vector<graphlib::Shape> generate_pool_2d_shapes()
{
    std::vector<graphlib::Shape> shapes;

    // 4D shapes: (N, C, H, W)
    auto shapes_range = shape_range({1, 1, 8, 8}, {2, 2, 16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    return shapes;
}

// Representative individual test shapes
std::vector<VecShapes> get_pool_1d_individual_shapes()
{
    return {
        VecShapes{{1, 1, 8}},   // Minimal 3D
        VecShapes{{1, 2, 16}},  // Small channels
        VecShapes{{2, 4, 32}},  // Multiple batch/channels
        VecShapes{{1, 8, 64}},  // Larger width
    };
}

std::vector<VecShapes> get_pool_2d_individual_shapes()
{
    return {
        VecShapes{{1, 1, 8, 8}},    // Minimal 4D
        VecShapes{{1, 2, 16, 16}},  // Small channels
        VecShapes{{2, 4, 32, 32}},  // Multiple batch/channels
        VecShapes{{1, 8, 28, 28}},  // Common CNN size
    };
}

// Individual test configurations for AvgPool1d
std::vector<OpTestParam> generate_avg_pool_1d_individual_params()
{
    std::vector<OpTestParam> params;
    auto shapes = get_pool_1d_individual_shapes();

    // Selected representative configurations
    std::vector<std::tuple<int, int, int, bool>> configs = {
        {8, 1, 0, false},  // Basic 8x1 no padding
        {8, 2, 1, false},  // 8x2 with padding
        {8, 2, 0, true},   // 8x2 ceil mode
        {8, 3, 2, true},   // 8x3 with dilation
    };

    for (const auto& shape_vec : shapes)
    {
        for (const auto& [kernel_size, stride, padding, ceil_mode] : configs)
        {
            if (valid_avg_pool_1d_inputs(shape_vec, kernel_size, stride, padding))
            {
                tt::ops::Op op(
                    tt::ops::OpType::AvgPool1d,
                    {{"kernel_size", kernel_size},
                     {"stride", stride},
                     {"dilation", 1},
                     {"ceil_mode", ceil_mode},
                     {"padding_left", padding},
                     {"padding_right", padding},
                     {"count_include_pad", true}});
                params.emplace_back(op, shape_vec);
            }
        }
    }

    return params;
}

// Comprehensive sweep test configurations for AvgPool1d
std::vector<OpTestParam> generate_avg_pool_1d_sweep_params()
{
    std::vector<OpTestParam> params;
    auto shapes = generate_pool_1d_shapes();
    int stride = 1;
    int padding = 0;
    bool ceil_mode = false;

    // Comprehensive parameter combinations
    for (int kernel_size : {8, 16, 32})
    {
        for (const auto& shape : shapes)
        {
            std::vector<graphlib::Shape> shape_vec = {shape};
            if (valid_avg_pool_1d_inputs(shape_vec, kernel_size, stride, padding))
            {
                tt::ops::Op op(
                    tt::ops::OpType::AvgPool1d,
                    {{"kernel_size", kernel_size},
                     {"stride", stride},
                     {"dilation", 1},
                     {"ceil_mode", ceil_mode},
                     {"padding_left", padding},
                     {"padding_right", padding},
                     {"count_include_pad", true}});
                params.emplace_back(op, shape_vec);
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    AvgPool1dIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_avg_pool_1d_individual_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    AvgPool1dSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_avg_pool_1d_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Individual test configurations for AvgPool2d
std::vector<OpTestParam> generate_avg_pool_2d_individual_params()
{
    std::vector<OpTestParam> params;
    auto shapes = get_pool_2d_individual_shapes();

    // Selected representative configurations
    std::vector<std::tuple<int, int, int, int, int, bool, bool>> configs = {
        {2, 2, 1, 1, 0, false, false},  // Basic 2x2 stride 1
        {3, 3, 2, 2, 1, false, false},  // 3x3 stride 2 with padding
        {2, 2, 1, 1, 0, true, false},   // 2x2 ceil mode channel_last
        {4, 4, 2, 2, 1, true, false},   // Complex config channel_last
    };

    for (const auto& shape_vec : shapes)
    {
        for (const auto& [kernel_h, kernel_w, stride_h, stride_w, padding, ceil_mode, channel_last] : configs)
        {
            if (valid_pool_2d_inputs(
                    shape_vec, kernel_h, kernel_w, stride_h, stride_w, padding, padding, 1, 1, channel_last))
            {
                tt::ops::Op op(
                    tt::ops::OpType::AvgPool2d,
                    {{"kernel_height", kernel_h},
                     {"kernel_width", kernel_w},
                     {"stride_height", stride_h},
                     {"stride_width", stride_w},
                     {"dilation_height", 1},
                     {"dilation_width", 1},
                     {"ceil_mode", ceil_mode},
                     {"padding_left", padding},
                     {"padding_right", padding},
                     {"padding_top", padding},
                     {"padding_bottom", padding},
                     {"count_include_pad", true},
                     {"channel_last", channel_last}});
                params.emplace_back(op, shape_vec);
            }
        }
    }

    return params;
}

// Comprehensive sweep test configurations for AvgPool2d
std::vector<OpTestParam> generate_avg_pool_2d_sweep_params()
{
    std::vector<OpTestParam> params;
    auto shapes = generate_pool_2d_shapes();
    int stride_h = 1;
    int stride_w = 1;
    int padding = 0;
    bool ceil_mode = false;
    bool channel_last = false;

    // Comprehensive parameter combinations
    for (int kernel_h : {2, 3})
    {
        for (int kernel_w : {2, 3})
        {
            for (const auto& shape : shapes)
            {
                std::vector<graphlib::Shape> shape_vec = {shape};
                if (valid_pool_2d_inputs(
                        shape_vec, kernel_h, kernel_w, stride_h, stride_w, padding, padding, 1, 1, channel_last))
                {
                    tt::ops::Op op(
                        tt::ops::OpType::AvgPool2d,
                        {{"kernel_height", kernel_h},
                         {"kernel_width", kernel_w},
                         {"stride_height", stride_h},
                         {"stride_width", stride_w},
                         {"dilation_height", 1},
                         {"dilation_width", 1},
                         {"ceil_mode", ceil_mode},
                         {"padding_left", padding},
                         {"padding_right", padding},
                         {"padding_top", padding},
                         {"padding_bottom", padding},
                         {"count_include_pad", true},
                         {"channel_last", channel_last}});
                    params.emplace_back(op, shape_vec);
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    AvgPool2dIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_avg_pool_2d_individual_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    AvgPool2dSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_avg_pool_2d_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Reuse the same logic as avg_pool_1d but with MaxPool1d operations
std::vector<OpTestParam> generate_max_pool_1d_individual_params()
{
    std::vector<OpTestParam> params;
    auto shapes = get_pool_1d_individual_shapes();

    // Same configurations as avg pool
    std::vector<std::tuple<int, int, int, bool>> configs = {
        {2, 1, 0, false},  // Basic 2x1 no padding
        {3, 2, 1, false},  // 3x2 with padding
        {4, 2, 0, true},   // 4x2 ceil mode
        {5, 3, 2, true},   // Complex config
    };

    for (const auto& shape_vec : shapes)
    {
        for (const auto& [kernel_size, stride, padding, ceil_mode] : configs)
        {
            if (valid_pool_1d_inputs(shape_vec, kernel_size, stride, padding))
            {
                tt::ops::Op op(
                    tt::ops::OpType::MaxPool1d,
                    {{"kernel_size", kernel_size},
                     {"stride", stride},
                     {"dilation", 1},
                     {"ceil_mode", ceil_mode},
                     {"padding", padding}});
                params.emplace_back(op, shape_vec);
            }
        }
    }

    return params;
}

std::vector<OpTestParam> generate_max_pool_1d_sweep_params()
{
    std::vector<OpTestParam> params;
    auto shapes = generate_pool_1d_shapes();
    int stride = 1;
    int padding = 0;
    bool ceil_mode = false;

    for (int kernel_size : {2, 3, 4})
    {
        for (const auto& shape : shapes)
        {
            std::vector<graphlib::Shape> shape_vec = {shape};
            if (valid_pool_1d_inputs(shape_vec, kernel_size, stride, padding))
            {
                tt::ops::Op op(
                    tt::ops::OpType::MaxPool1d,
                    {{"kernel_size", kernel_size},
                     {"stride", stride},
                     {"dilation", 1},
                     {"ceil_mode", ceil_mode},
                     {"padding", padding}});
                params.emplace_back(op, shape_vec);
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MaxPool1dIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_max_pool_1d_individual_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    MaxPool1dSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_max_pool_1d_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Reuse the same logic as avg_pool_2d but with MaxPool2d operations
std::vector<OpTestParam> generate_max_pool_2d_individual_params()
{
    std::vector<OpTestParam> params;
    auto shapes = get_pool_2d_individual_shapes();

    // Same configurations as avg pool but without count_include_pad
    std::vector<std::tuple<int, int, int, int, int, bool, bool>> configs = {
        {2, 2, 1, 1, 0, false, false},  // Basic 2x2 stride 1
        {3, 3, 2, 2, 1, false, false},  // 3x3 stride 2 with padding
        {2, 2, 1, 1, 0, true, true},    // 2x2 ceil mode channel_last
        {4, 4, 2, 2, 1, true, true},    // Complex config channel_last
    };

    for (const auto& shape_vec : shapes)
    {
        for (const auto& [kernel_h, kernel_w, stride_h, stride_w, padding, ceil_mode, channel_last] : configs)
        {
            if (valid_pool_2d_inputs(
                    shape_vec, kernel_h, kernel_w, stride_h, stride_w, padding, padding, 1, 1, channel_last))
            {
                tt::ops::Op op(
                    tt::ops::OpType::MaxPool2d,
                    {{"kernel_height", kernel_h},
                     {"kernel_width", kernel_w},
                     {"stride_height", stride_h},
                     {"stride_width", stride_w},
                     {"dilation_height", 1},
                     {"dilation_width", 1},
                     {"ceil_mode", ceil_mode},
                     {"padding_left", padding},
                     {"padding_right", padding},
                     {"padding_top", padding},
                     {"padding_bottom", padding},
                     {"channel_last", channel_last}});
                params.emplace_back(op, shape_vec);
            }
        }
    }

    return params;
}

std::vector<OpTestParam> generate_max_pool_2d_sweep_params()
{
    std::vector<OpTestParam> params;
    auto shapes = generate_pool_2d_shapes();
    int stride_h = 1;
    int stride_w = 1;
    int padding = 0;
    bool ceil_mode = false;
    bool channel_last = false;

    for (int kernel_h : {2, 3})
    {
        for (int kernel_w : {2, 3})
        {
            for (const auto& shape : shapes)
            {
                std::vector<graphlib::Shape> shape_vec = {shape};
                if (valid_pool_2d_inputs(
                        shape_vec, kernel_h, kernel_w, stride_h, stride_w, padding, padding, 1, 1, channel_last))
                {
                    tt::ops::Op op(
                        tt::ops::OpType::MaxPool2d,
                        {{"kernel_height", kernel_h},
                         {"kernel_width", kernel_w},
                         {"stride_height", stride_h},
                         {"stride_width", stride_w},
                         {"dilation_height", 1},
                         {"dilation_width", 1},
                         {"ceil_mode", ceil_mode},
                         {"padding_left", padding},
                         {"padding_right", padding},
                         {"padding_top", padding},
                         {"padding_bottom", padding},
                         {"channel_last", channel_last}});
                    params.emplace_back(op, shape_vec);
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MaxPool2dIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_max_pool_2d_individual_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    MaxPool2dSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_max_pool_2d_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::pooling
