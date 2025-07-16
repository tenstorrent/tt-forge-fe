// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::eltwise_binary
{

bool valid_inputs(const graphlib::Shape& in1, const graphlib::Shape& in2)
{
    return in1 == in2 || graphlib::can_be_broadcasted(in1, in2);
}

std::vector<std::vector<graphlib::Shape>> generate_input_shapes()
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

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        input_shapes.push_back({shapes[i], shapes[i]});
        for (size_t j = i + 1; j < shapes.size(); ++j)
        {
            if (valid_inputs(shapes[i], shapes[j]))
            {
                input_shapes.push_back({shapes[i], shapes[j]});
                input_shapes.push_back({shapes[j], shapes[i]});
            }
        }
    }

    return input_shapes;
}

INSTANTIATE_TEST_SUITE_P(
    BinaryEltwiseOpsIndividual,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(
                tt::ops::OpType::Add, tt::ops::OpType::Multiply, tt::ops::OpType::Divide, tt::ops::OpType::Subtract),
            testing::Values(
                VecShapes{{1, 1, 1, 32}, {1, 1, 1, 32}},
                VecShapes{{1, 1, 32, 1}, {1, 1, 32, 1}},
                VecShapes{{1, 32, 1, 1}, {1, 32, 1, 1}},
                VecShapes{{32, 1, 1, 1}, {32, 1, 1, 1}},
                VecShapes{{1, 2, 3, 4}, {1, 2, 3, 4}},
                VecShapes{{2, 3, 4}, {2, 3, 4}},
                VecShapes{{3, 4}, {3, 4}},
                VecShapes{{4}, {4}},
                VecShapes{{1}, {1}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    BinaryEltwiseOpsSweep,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(
                tt::ops::OpType::Add, tt::ops::OpType::Multiply, tt::ops::OpType::Divide, tt::ops::OpType::Subtract),
            testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::eltwise_binary

namespace tt::test::ops::eltwise_unary
{

std::vector<std::vector<graphlib::Shape>> generate_input_shapes()
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

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        input_shapes.push_back({shapes[i]});
    }

    return input_shapes;
}

INSTANTIATE_TEST_SUITE_P(
    UnaryEltwiseOpsIndividual,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(tt::ops::OpType::Abs),
            testing::Values(
                VecShapes{{1, 1, 1, 32}},
                VecShapes{{1, 1, 32, 1}},
                VecShapes{{1, 32, 1, 1}},
                VecShapes{{32, 1, 1, 1}},
                VecShapes{{1, 2, 3, 4}},
                VecShapes{{2, 3, 4}},
                VecShapes{{3, 4}},
                VecShapes{{4}},
                VecShapes{{1}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    UnaryEltwiseOpsSweep,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(tt::ops::OpType::Abs, tt::ops::OpType::Sine, tt::ops::OpType::Cosine),
            testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::eltwise_unary
