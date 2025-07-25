// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::adv_index
{

bool valid_inputs(const std::vector<graphlib::Shape>& shapes, int dim)
{
    const graphlib::Shape& data_shape = shapes[0];
    const graphlib::Shape& indices_shape = shapes[1];
    if (dim < 0)
    {
        dim += data_shape.size();
    }

    // Check if indices shape don't exceed the dimension size
    uint32_t dim_size = data_shape[dim];
    uint32_t indeces_size = 1;
    for (size_t i = 0; i < indices_shape.size(); ++i)
    {
        indeces_size *= indices_shape[i];
    }

    return indeces_size <= dim_size;
}

std::vector<OpTestParam> generate_adv_index_test_params()
{
    // Representative input shapes covering 1D-4D tensors
    auto input_shapes = {
        graphlib::Shape{8},           // 1D
        graphlib::Shape{4, 8},        // 2D
        graphlib::Shape{2, 4, 8},     // 3D
        graphlib::Shape{2, 3, 4, 8},  // 4D
    };

    // Representative index shapes covering 1D and 2D indices
    auto index_shapes = {
        graphlib::Shape{2},     // 1D indices
        graphlib::Shape{2, 1},  // 2D indices
        graphlib::Shape{1, 2},  // 2D indices (different layout)
    };

    std::vector<OpTestParam> params;

    // Generate all valid combinations
    for (const auto& data_shape : input_shapes)
    {
        for (const auto& indices_shape : index_shapes)
        {
            // Test positive dimensions
            for (int dim = 0; dim < static_cast<int>(data_shape.size()); ++dim)
            {
                std::vector<graphlib::Shape> shapes = {data_shape, indices_shape};
                if (valid_inputs(shapes, dim))
                {
                    tt::ops::Op op(tt::ops::OpType::AdvIndex, {{"dim", dim}});
                    params.emplace_back(op, shapes);
                }
            }

            // Test negative dimensions
            for (int neg_dim = -1; neg_dim >= -static_cast<int>(data_shape.size()); --neg_dim)
            {
                std::vector<graphlib::Shape> shapes = {data_shape, indices_shape};
                if (valid_inputs(shapes, neg_dim))
                {
                    tt::ops::Op op(tt::ops::OpType::AdvIndex, {{"dim", neg_dim}});
                    params.emplace_back(op, shapes);
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    AdvIndexOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_adv_index_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

std::vector<graphlib::Shape> generate_input_shapes()
{
    std::vector<graphlib::Shape> shapes;

    // Generate systematic shape ranges like eltwise/reduce tests
    auto shapes_range = shape_range({1}, {8});  // 1D: [1] to [8]
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {8, 8});  // 2D: [1,1] to [8,8]
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {4, 4, 4});  // 3D: [1,1,1] to [4,4,4]
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    return shapes;
}

std::vector<OpTestParam> generate_adv_index_sweep_params()
{
    auto data_shapes = generate_input_shapes();

    // Simpler index shapes for sweep tests
    auto index_shapes = {
        graphlib::Shape{1},     // Small 1D indices
        graphlib::Shape{2},     // Medium 1D indices
        graphlib::Shape{1, 2},  // Small 2D indices
        graphlib::Shape{2, 1},  // Medium 2D indices
    };

    std::vector<OpTestParam> params;

    // Generate all valid combinations for sweep
    for (const auto& data_shape : data_shapes)
    {
        for (const auto& indices_shape : index_shapes)
        {
            // Test a subset of dimensions to keep test count reasonable
            std::vector<int> test_dims = {0, -1};  // First and last dimensions

            // Add middle dimension for 3D+ tensors
            if (data_shape.size() >= 3)
            {
                test_dims.push_back(1);
            }

            for (int dim : test_dims)
            {
                std::vector<graphlib::Shape> shapes = {data_shape, indices_shape};
                if (valid_inputs(shapes, dim))
                {
                    tt::ops::Op op(tt::ops::OpType::AdvIndex, {{"dim", dim}});
                    params.emplace_back(op, shapes);
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    AdvIndexOpsSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_adv_index_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::adv_index
