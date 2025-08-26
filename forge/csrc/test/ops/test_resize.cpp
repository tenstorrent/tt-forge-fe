// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::resize
{

// Helper function to create resize2d operations
tt::ops::Op create_resize2d(
    std::vector<int> sizes, std::string mode = "nearest", bool align_corners = false, bool channel_last = false)
{
    return tt::ops::Op(
        tt::ops::OpType::Resize2d,
        {{"sizes", sizes}, {"mode", mode}, {"align_corners", align_corners}, {"channel_last", channel_last}});
}

// Helper function to create resize1d operations
tt::ops::Op create_resize1d(
    int size, std::string mode = "nearest", bool align_corners = false, bool channel_last = false)
{
    return tt::ops::Op(
        tt::ops::OpType::Resize1d,
        {{"size", size}, {"mode", mode}, {"align_corners", align_corners}, {"channel_last", channel_last}});
}

// Helper function to create upsample2d operations
tt::ops::Op create_upsample2d(std::vector<int> scale_factor, std::string mode = "nearest", bool channel_last = false)
{
    return tt::ops::Op(
        tt::ops::OpType::Upsample2d, {{"scale_factor", scale_factor}, {"mode", mode}, {"channel_last", channel_last}});
}

// Validation functions

bool valid_resize2d_inputs(
    const std::vector<graphlib::Shape>& shapes,
    const std::vector<int>& sizes,
    bool align_corners = false,
    bool channel_last = false)
{
    // Check input dimensions
    uint32_t input_h = channel_last ? shapes[0][shapes[0].size() - 3] : shapes[0][shapes[0].size() - 2];
    uint32_t input_w = channel_last ? shapes[0][shapes[0].size() - 2] : shapes[0][shapes[0].size() - 1];

    // For integer scale factors, check divisibility
    bool upsample_h = sizes[0] >= static_cast<int>(input_h);
    bool upsample_w = sizes[1] >= static_cast<int>(input_w);

    if (upsample_h && upsample_w)
    {
        if (align_corners)
            return false;  // align_corners argument not supported in post init decomposition of resize2d to upsample2d

        // Upsampling: check integer scale factors
        return (sizes[0] % input_h == 0) && (sizes[1] % input_w == 0);
    }
    else if (!upsample_h && !upsample_w)
    {
        // TODO: Implement downsampling validation
        return false;
        // Downsampling: check integer scale factors
        // return (input_h % sizes[0] == 0) && (input_w % sizes[1] == 0) &&
        //        (input_h / sizes[0] == input_w / sizes[1]);  // Same scale factor
    }

    return false;  // Mixed up/down scaling not supported
}

bool valid_resize1d_inputs(
    const std::vector<graphlib::Shape>& shapes, int size, bool align_corners = false, bool channel_last = false)
{
    // Check input dimensions
    uint32_t input_w = channel_last ? shapes[0][shapes[0].size() - 2] : shapes[0][shapes[0].size() - 1];

    // For integer scale factors, check divisibility
    bool is_upsample = size >= static_cast<int>(input_w);

    if (is_upsample)
    {
        if (align_corners)
            return false;  // align_corners argument not supported in post init decomposition of resize1d to upsample2d

        // Upsampling: check integer scale factors
        return (size % input_w == 0);
    }

    return false;  // Downsampling not supported
}

// Test shape generators

std::vector<graphlib::Shape> generate_resize2d_shapes()
{
    std::vector<graphlib::Shape> shapes;

    // 4D shapes: (N, C, H, W)
    auto shapes_range = shape_range({1, 1, 4, 4}, {2, 4, 16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    return shapes;
}

std::vector<graphlib::Shape> generate_resize1d_shapes()
{
    std::vector<graphlib::Shape> shapes;

    // 3D shapes: (N, C, W)
    auto shapes_range = shape_range({1, 1, 4}, {2, 4, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    return shapes;
}

std::vector<graphlib::Shape> generate_upsample2d_shapes()
{
    std::vector<graphlib::Shape> shapes;

    // 4D shapes: (N, C, H, W)
    auto shapes_range = shape_range({1, 1, 4, 4}, {2, 4, 8, 8});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    return shapes;
}

// Individual test shapes

std::vector<VecShapes> get_resize2d_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 4, 4}},    // Minimal 4D
        VecShapes{{1, 2, 8, 8}},    // Small channels
        VecShapes{{2, 4, 4, 4}},    // Multiple batch/channels
        VecShapes{{1, 3, 8, 8}},    // RGB-like
        VecShapes{{1, 1, 16, 16}},  // Larger spatial
        VecShapes{{1, 3, 9, 16}},   // different height and width
    };
}

std::vector<VecShapes> get_resize1d_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 4}},   // Minimal 3D
        VecShapes{{1, 2, 8}},   // Small channels
        VecShapes{{2, 4, 4}},   // Multiple batch/channels
        VecShapes{{1, 3, 8}},   // RGB-like
        VecShapes{{1, 1, 16}},  // Larger spatial
    };
}

std::vector<VecShapes> get_upsample2d_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 4, 4}},   // Minimal 4D
        VecShapes{{1, 2, 4, 4}},   // Small channels
        VecShapes{{2, 4, 4, 4}},   // Multiple batch/channels
        VecShapes{{1, 3, 8, 8}},   // RGB-like
        VecShapes{{1, 3, 9, 16}},  // different height and width
    };
}

// Resize2D operations
std::vector<tt::ops::Op> get_resize2d_ops()
{
    std::vector<tt::ops::Op> ops;

    // Different target sizes and modes
    std::vector<std::vector<int>> sizes = {{8, 8}, {16, 16}, {4, 4}, {12, 12}, {8, 12}, {9, 16}, {15, 21}};
    std::vector<std::string> modes = {"nearest", "bilinear"};
    std::vector<bool> channel_last_options = {false, true};
    std::vector<bool> align_corners_options = {false, true};

    for (const auto& size : sizes)
    {
        for (const std::string& mode : modes)
        {
            for (bool channel_last : channel_last_options)
            {
                for (bool align_corners : align_corners_options)
                {
                    ops.push_back(create_resize2d(size, mode, align_corners, channel_last));
                }
            }
        }
    }

    return ops;
}

// Resize1D operations
std::vector<tt::ops::Op> get_resize1d_ops()
{
    std::vector<tt::ops::Op> ops;

    // Different target sizes and modes
    std::vector<int> sizes = {8, 9, 12, 16};
    std::vector<std::string> modes = {"nearest", "linear"};
    std::vector<bool> channel_last_options = {false, true};
    std::vector<bool> align_corners_options = {false, true};

    for (const auto& size : sizes)
    {
        for (const std::string& mode : modes)
        {
            for (bool channel_last : channel_last_options)
            {
                for (bool align_corners : align_corners_options)
                {
                    ops.push_back(create_resize1d(size, mode, align_corners, channel_last));
                }
            }
        }
    }

    return ops;
}

// Upsample2D operations
std::vector<tt::ops::Op> get_upsample2d_ops()
{
    std::vector<tt::ops::Op> ops;

    std::vector<std::vector<int>> scale_factors = {{2, 2}, {2, 3}, {8, 4}, {3, 9}};
    std::vector<std::string> modes = {"nearest", "bilinear"};
    std::vector<bool> channel_last_options = {false, true};

    for (const auto& scale_factor : scale_factors)
    {
        for (const std::string& mode : modes)
        {
            for (bool channel_last : channel_last_options)
            {
                ops.push_back(create_upsample2d(scale_factor, mode, channel_last));
            }
        }
    }

    return ops;
}

// Generate individual test combinations for resize2d
std::vector<OpTestParam> generate_resize2d_individual_combinations()
{
    std::vector<OpTestParam> combinations;
    auto ops = get_resize2d_ops();
    auto shapes = get_resize2d_individual_test_shapes();

    for (const auto& op : ops)
    {
        auto sizes = op.attr_as<std::vector<int>>("sizes");
        bool channel_last = op.attr_as<bool>("channel_last");
        bool align_corners = op.attr_as<bool>("align_corners");

        for (const auto& shape_vec : shapes)
        {
            if (valid_resize2d_inputs(shape_vec, sizes, align_corners, channel_last))
            {
                combinations.push_back({op, shape_vec});
            }
        }
    }

    return combinations;
}

// Generate individual test combinations for resize1d
std::vector<OpTestParam> generate_resize1d_individual_combinations()
{
    std::vector<OpTestParam> combinations;
    auto ops = get_resize1d_ops();
    auto shapes = get_resize1d_individual_test_shapes();

    for (const auto& op : ops)
    {
        auto size = op.attr_as<int>("size");
        bool channel_last = op.attr_as<bool>("channel_last");
        bool align_corners = op.attr_as<bool>("align_corners");

        for (const auto& shape_vec : shapes)
        {
            if (valid_resize1d_inputs(shape_vec, size, align_corners, channel_last))
            {
                combinations.push_back({op, shape_vec});
            }
        }
    }

    return combinations;
}

// Generate valid test combinations for resize2d sweep
std::vector<OpTestParam> generate_resize2d_sweep_combinations()
{
    std::vector<OpTestParam> valid_combinations;
    auto ops = get_resize2d_ops();
    auto shapes = generate_resize2d_shapes();

    for (const auto& op : ops)
    {
        auto sizes = op.attr_as<std::vector<int>>("sizes");
        bool channel_last = op.attr_as<bool>("channel_last");
        bool align_corners = op.attr_as<bool>("align_corners");

        for (const auto& shape : shapes)
        {
            VecShapes shape_vec = {shape};
            if (valid_resize2d_inputs(shape_vec, sizes, align_corners, channel_last))
            {
                valid_combinations.push_back({op, shape_vec});
            }
        }
    }

    return valid_combinations;
}

// Generate valid test combinations for resize1d sweep
std::vector<OpTestParam> generate_resize1d_sweep_combinations()
{
    std::vector<OpTestParam> valid_combinations;
    auto ops = get_resize1d_ops();
    auto shapes = generate_resize1d_shapes();

    for (const auto& op : ops)
    {
        auto size = op.attr_as<int>("size");
        bool channel_last = op.attr_as<bool>("channel_last");
        bool align_corners = op.attr_as<bool>("align_corners");

        for (const auto& shape : shapes)
        {
            VecShapes shape_vec = {shape};
            if (valid_resize1d_inputs(shape_vec, size, align_corners, channel_last))
            {
                valid_combinations.push_back({op, shape_vec});
            }
        }
    }

    return valid_combinations;
}

// Generate individual test combinations for upsample2d
std::vector<OpTestParam> generate_upsample2d_individual_combinations()
{
    std::vector<OpTestParam> combinations;
    auto ops = get_upsample2d_ops();
    auto shapes = get_upsample2d_individual_test_shapes();

    for (const auto& op : ops)
    {
        for (const auto& shape_vec : shapes)
        {
            combinations.push_back({op, shape_vec});
        }
    }

    return combinations;
}

// Generate sweep test combinations for upsample2d
std::vector<OpTestParam> generate_upsample2d_sweep_combinations()
{
    std::vector<OpTestParam> valid_combinations;
    auto ops = get_upsample2d_ops();
    auto shapes = generate_upsample2d_shapes();

    for (const auto& op : ops)
    {
        for (const auto& shape : shapes)
        {
            VecShapes shape_vec = {shape};
            valid_combinations.push_back({op, shape_vec});
        }
    }

    return valid_combinations;
}

// Test instantiations for Resize2D
INSTANTIATE_TEST_SUITE_P(
    Resize2DIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_resize2d_individual_combinations()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    Resize2DSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_resize2d_sweep_combinations()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Test instantiations for Resize1D
INSTANTIATE_TEST_SUITE_P(
    Resize1DIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_resize1d_individual_combinations()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    Resize1DSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_resize1d_sweep_combinations()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Test instantiations for Upsample2D
INSTANTIATE_TEST_SUITE_P(
    Upsample2DIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_upsample2d_individual_combinations()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    Upsample2DSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_upsample2d_sweep_combinations()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::resize
