// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::log_softmax
{

tt::ops::Op get_log_softmax_op() { return tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 0}, {"stable", true}}); }

std::vector<OpTestParam> generate_individual_tests()
{
    std::vector<OpTestParam> params;

    // Test different dimensions with specific shapes

    // Dimension 0 tests
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 0}, {"stable", true}}),
        std::vector{graphlib::Shape{2, 3, 4}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 0}, {"stable", false}}),
        std::vector{graphlib::Shape{5, 6, 7}});

    // Dimension 1 tests
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 1}, {"stable", true}}),
        std::vector{graphlib::Shape{2, 3, 4}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 1}, {"stable", false}}),
        std::vector{graphlib::Shape{5, 6, 7}});

    // Dimension 2 tests
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 2}, {"stable", true}}),
        std::vector{graphlib::Shape{2, 3, 4}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 2}, {"stable", false}}),
        std::vector{graphlib::Shape{5, 6, 7}});

    // Negative dimension tests
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", -1}, {"stable", true}}),
        std::vector{graphlib::Shape{2, 3, 4}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", -1}, {"stable", false}}), std::vector{graphlib::Shape{5, 6}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", -2}, {"stable", true}}),
        std::vector{graphlib::Shape{2, 3, 4}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", -2}, {"stable", false}}),
        std::vector{graphlib::Shape{5, 6, 7}});

    // Edge cases
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 0}, {"stable", true}}), std::vector{graphlib::Shape{1}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 0}, {"stable", false}}), std::vector{graphlib::Shape{32}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 1}, {"stable", true}}), std::vector{graphlib::Shape{1, 32}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", -1}, {"stable", false}}),
        std::vector{graphlib::Shape{1, 2, 3, 4}});

    // 4D tensor tests
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", 3}, {"stable", true}}),
        std::vector{graphlib::Shape{1, 2, 3, 4}});
    params.emplace_back(
        tt::ops::Op(tt::ops::OpType::LogSoftmax, {{"dim", -4}, {"stable", false}}),
        std::vector{graphlib::Shape{1, 2, 3, 4}});

    return params;
}

std::vector<std::vector<graphlib::Shape>> generate_input_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range({1}, {8});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {8, 8});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {4, 4, 4});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1, 1}, {3, 3, 3, 3});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    for (const auto& shape : shapes)
    {
        input_shapes.push_back({shape});
    }

    return input_shapes;
}

bool valid_inputs(const std::vector<graphlib::Shape>& shapes, int dim)
{
    if (shapes.size() != 1)
        return false;

    const auto& shape = shapes[0];
    int normalized_dim = dim;
    if (dim < 0)
    {
        normalized_dim = static_cast<int>(shape.size()) + dim;
    }

    return normalized_dim >= 0 && normalized_dim < static_cast<int>(shape.size());
}

std::vector<OpTestParam> generate_valid_log_softmax_params()
{
    std::vector<OpTestParam> valid_params;
    auto input_shapes_vec = generate_input_shapes();

    for (const auto& shapes : input_shapes_vec)
    {
        const auto& shape = shapes[0];

        // Test different dimensions for this shape
        for (int dim = 0; dim < static_cast<int>(shape.size()); ++dim)
        {
            for (bool stable : {true, false})
            {
                tt::ops::Op op(tt::ops::OpType::LogSoftmax, {{"dim", dim}, {"stable", stable}});
                valid_params.emplace_back(op, shapes);
            }
        }

        // Test negative dimensions
        for (int dim = -1; dim >= -static_cast<int>(shape.size()); --dim)
        {
            for (bool stable : {true, false})
            {
                tt::ops::Op op(tt::ops::OpType::LogSoftmax, {{"dim", dim}, {"stable", stable}});
                valid_params.emplace_back(op, shapes);
            }
        }
    }

    return valid_params;
}

// Individual test cases for specific important scenarios
INSTANTIATE_TEST_SUITE_P(
    LogSoftmaxIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_individual_tests()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Comprehensive sweep test
INSTANTIATE_TEST_SUITE_P(
    LogSoftmaxSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_valid_log_softmax_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::log_softmax
