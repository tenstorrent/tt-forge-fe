// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::reduce
{

std::vector<OpTestParam> generate_reduce_test_params()
{
    std::vector<tt::ops::OpType> reduce_ops = {
        tt::ops::OpType::ReduceAvg,
        tt::ops::OpType::ReduceMax,
        tt::ops::OpType::ReduceSum,
    };

    auto input_shapes = {
        graphlib::Shape{1, 1, 1, 32},
        graphlib::Shape{1, 1, 32, 1},
        graphlib::Shape{1, 32, 1, 1},
        graphlib::Shape{32, 1, 1, 1},
        graphlib::Shape{1, 2, 3, 4},
        graphlib::Shape{2, 3, 4},
        graphlib::Shape{3, 4},
        graphlib::Shape{4},
        graphlib::Shape{1}};

    std::vector<OpTestParam> params;

    //  Generate test parameters for every possible combination of reduce op, input shape, dimension, and keep_dim
    //  value.
    for (const auto& op_type : reduce_ops)
    {
        for (const auto& shape : input_shapes)
        {
            for (int dim = 0; dim < static_cast<int>(shape.size()); ++dim)
            {
                for (bool keep_dim : {true, false})
                {
                    std::vector<int> dim_arg = {dim};
                    tt::ops::Op op(op_type, {{"dim_arg", dim_arg}, {"keep_dim", keep_dim}});
                    params.emplace_back(op, std::vector{shape});
                }
            }

            for (int neg_dim = -1; neg_dim >= -static_cast<int>(shape.size()); --neg_dim)
            {
                for (bool keep_dim : {true, false})
                {
                    std::vector<int> dim_arg = {neg_dim};
                    tt::ops::Op op(op_type, {{"dim_arg", dim_arg}, {"keep_dim", keep_dim}});
                    params.emplace_back(op, std::vector{shape});
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    ReduceOpsIndividual,
    SimpleOpTest,
    testing::ValuesIn(generate_reduce_test_params()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

std::vector<graphlib::Shape> generate_input_shapes()
{
    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range({1}, {16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {5, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1, 1}, {5, 1, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    return shapes;
}

std::vector<OpTestParam> generate_reduce_sweep_params()
{
    std::vector<tt::ops::OpType> reduce_ops = {
        tt::ops::OpType::ReduceAvg,
        tt::ops::OpType::ReduceMax,
        tt::ops::OpType::ReduceSum,
    };

    auto input_shapes = generate_input_shapes();

    std::vector<OpTestParam> params;

    // Generate test parameters for every possible combination of reduce op, input shape, dimension, and keep_dim.
    for (const auto& op_type : reduce_ops)
    {
        for (const auto& shape : input_shapes)
        {
            std::vector<int> test_dims;

            for (int dim = 0; dim < static_cast<int>(shape.size()); ++dim)
            {
                test_dims.push_back(dim);
                test_dims.push_back(-dim - 1);
            }

            for (int dim : test_dims)
            {
                for (bool keep_dim : {true, false})
                {
                    std::vector<int> dim_arg = {dim};
                    tt::ops::Op op(op_type, {{"dim_arg", dim_arg}, {"keep_dim", keep_dim}});
                    params.emplace_back(op, std::vector{shape});
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    ReduceOpsSweep,
    SimpleOpTest,
    testing::ValuesIn(generate_reduce_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::reduce
