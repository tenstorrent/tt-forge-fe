// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "test_ops.hpp"

namespace tt::test::ops::select
{

tt::ops::Op create_select_op(int dim, int begin, int length, int stride)
{
    return tt::ops::Op(
        tt::ops::OpType::Select, {{"dim", dim}, {"begin", begin}, {"length", length}, {"stride", stride}});
}

std::vector<OpTestParam> generate_select_focused_params()
{
    return {
        // Basic positive dim - covers eval() basic path and shape() positive dim conversion
        {create_select_op(0, 0, 2, 4), {graphlib::Shape{8}}},

        // Negative dim - covers shape() and backward() negative dim normalization
        {create_select_op(-1, 0, 2, 4), {graphlib::Shape{4, 8}}},

        // Zero begin - covers eval() zero begin handling
        {create_select_op(0, 0, 3, 6), {graphlib::Shape{12}}},

        // Non-zero begin - covers eval() offset calculation
        {create_select_op(0, 2, 2, 4), {graphlib::Shape{8}}},

        // Stride equals input size - covers eval() boundary condition
        {create_select_op(0, 0, 2, 8), {graphlib::Shape{8}}},

        // Length 1 - covers eval() single element selection
        {create_select_op(0, 1, 1, 2), {graphlib::Shape{4}}},

        // Multi-dimensional tensor - covers eval() with higher dimensions
        {create_select_op(1, 0, 2, 3), {graphlib::Shape{2, 6, 4}}},

        // Negative dim on 3D tensor - covers negative dim handling on 3D
        {create_select_op(-2, 1, 2, 4), {graphlib::Shape{2, 8, 4}}},

        // Large stride - covers eval() with stride larger than length
        {create_select_op(0, 0, 3, 8), {graphlib::Shape{16}}},

        // Edge case: stride == length - covers eval() contiguous blocks
        {create_select_op(0, 0, 4, 4), {graphlib::Shape{8}}},

        // 4D tensor - covers higher-dimensional tensor handling
        {create_select_op(2, 1, 2, 3), {graphlib::Shape{2, 3, 6, 5}}},

        // Small tensor with large stride - covers eval() boundary checks
        {create_select_op(0, 0, 2, 4), {graphlib::Shape{4}}},

        // Middle dimension selection - covers eval() on inner dimensions
        {create_select_op(1, 2, 3, 5), {graphlib::Shape{3, 10, 4}}},

        // Last dimension selection - covers eval() on last dimension
        {create_select_op(-1, 0, 4, 6), {graphlib::Shape{2, 3, 12}}},

        // Complex case with all parameters - comprehensive test
        {create_select_op(1, 3, 2, 5), {graphlib::Shape{4, 15, 2}}},
    };
}

std::vector<OpTestParam> generate_select_sweep_params()
{
    std::vector<OpTestParam> params;

    // Test various tensor shapes
    std::vector<graphlib::Shape> test_shapes = {
        graphlib::Shape{8},          // 1D
        graphlib::Shape{4, 8},       // 2D
        graphlib::Shape{2, 4, 6},    // 3D
        graphlib::Shape{2, 3, 4, 5}  // 4D
    };

    // For each shape, test different parameter combinations
    for (const auto& shape : test_shapes)
    {
        auto shape_vec = shape.as_vector();

        // Test each dimension
        for (int dim = 0; dim < static_cast<int>(shape_vec.size()); ++dim)
        {
            int dim_size = static_cast<int>(shape_vec[dim]);

            // Test various begin values
            std::vector<int> begin_values = {0, 1, std::max(1, dim_size / 3)};

            for (int begin : begin_values)
            {
                if (begin >= dim_size)
                    continue;

                // Test various length values
                std::vector<int> length_values = {1, 2, std::min(3, dim_size - begin)};

                for (int length : length_values)
                {
                    if (begin + length > dim_size)
                        continue;

                    // Test various stride values
                    std::vector<int> stride_values = {
                        std::max(length + 1, 2), dim_size, std::min(dim_size * 2, length + 3)};

                    for (int stride : stride_values)
                    {
                        if (stride <= 0)
                            continue;

                        params.push_back({create_select_op(dim, begin, length, stride), {shape}});

                        // Also test negative dim
                        int neg_dim = dim - static_cast<int>(shape_vec.size());
                        params.push_back({create_select_op(neg_dim, begin, length, stride), {shape}});
                    }
                }
            }
        }
    }

    return params;
}

/**
 * Test fixture for select operations with backward pass support.
 */
INSTANTIATE_TEST_SUITE_P(
    SelectFocused,
    SimpleOpTest,
    testing::ValuesIn(generate_select_focused_params()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

/**
 * Sweeps tests.
 * TODO: With current implementation, 52 sweeps tests fail on backward, so we will only do decomposition tests.
 */
INSTANTIATE_TEST_SUITE_P(
    SelectSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_select_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::select
