// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::reduce
{
/**
 * Simple struct holding reduce op type and a flag representing whether dim_arg attribute is optional for that op.
 */
struct ReduceOpInfo
{
    tt::ops::OpType type;
    bool dim_arg_optional;
};

static std::vector<ReduceOpInfo> reduce_ops = {
    {tt::ops::OpType::ReduceAvg, false},
    {tt::ops::OpType::ReduceMax, false},
    {tt::ops::OpType::ReduceSum, false},
};

static std::vector<ReduceOpInfo> reduce_ops_no_backward = {
    {tt::ops::OpType::Argmax, true},
};

/*
 * Generates test parameters for every possible combination of reduce op, input shape, dimension, and keep_dim.
 */
std::vector<OpTestParam> generate_reduce_test_params(
    const std::vector<ReduceOpInfo>& ops, const std::vector<graphlib::Shape> input_shapes)
{
    std::vector<OpTestParam> params;

    for (const auto& reduce_op : ops)
    {
        for (const auto& shape : input_shapes)
        {
            for (bool keep_dim : {true, false})
            {
                if (reduce_op.dim_arg_optional)
                {
                    tt::ops::Op op(reduce_op.type, {{"keep_dim", keep_dim}});
                    params.emplace_back(op, std::vector{shape});
                }

                std::vector<int> dim_arg(1U);
                for (int dim = 0; dim < static_cast<int>(shape.size()); ++dim)
                {
                    dim_arg[0] = dim;
                    tt::ops::Op op1(reduce_op.type, {{"dim_arg", dim_arg}, {"keep_dim", keep_dim}});
                    params.emplace_back(op1, std::vector{shape});

                    dim_arg[0] = -dim - 1;
                    tt::ops::Op op2(reduce_op.type, {{"dim_arg", dim_arg}, {"keep_dim", keep_dim}});
                    params.emplace_back(op2, std::vector{shape});
                }
            }
        }
    }

    return params;
}

std::vector<graphlib::Shape> generate_reduce_test_shapes()
{
    return {
        graphlib::Shape{1, 1, 1, 32},
        graphlib::Shape{1, 1, 32, 1},
        graphlib::Shape{1, 32, 1, 1},
        graphlib::Shape{32, 1, 1, 1},
        graphlib::Shape{1, 2, 3, 4},
        graphlib::Shape{2, 3, 4},
        graphlib::Shape{3, 4},
        graphlib::Shape{4},
        graphlib::Shape{1}};
}

INSTANTIATE_TEST_SUITE_P(
    ReduceOpsIndividual,
    SimpleOpTest,
    testing::ValuesIn(generate_reduce_test_params(reduce_ops, generate_reduce_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    ReduceOpsNoBackwardIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_reduce_test_params(reduce_ops_no_backward, generate_reduce_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

std::vector<graphlib::Shape> generate_sweeps_input_shapes()
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

INSTANTIATE_TEST_SUITE_P(
    ReduceOpsSweep,
    SimpleOpTest,
    testing::ValuesIn(generate_reduce_test_params(reduce_ops, generate_sweeps_input_shapes())),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    ReduceOpsNoBackwardSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_reduce_test_params(reduce_ops_no_backward, generate_sweeps_input_shapes())),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::reduce
