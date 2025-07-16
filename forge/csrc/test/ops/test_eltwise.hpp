// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gtest/internal/gtest-param-util.h>
#include <torch/torch.h>

#include <array>
#include <initializer_list>
#include <vector>

#include "graph_lib/python_bindings.hpp"
#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::eltwise_binary
{

template <size_t N>
std::vector<graphlib::Shape> shape_range(
    std::array<uint32_t, N> min_shape, std::array<uint32_t, N> max_shape, size_t idx = 0)
{
    std::vector<graphlib::Shape> shapes;
    for (uint32_t val_dim = min_shape[idx]; val_dim <= max_shape[idx]; ++val_dim)
    {
        min_shape[idx] = val_dim;
        if (idx == N - 1)
        {
            std::vector<uint32_t> shape_vec(min_shape.begin(), min_shape.end());
            shapes.push_back(graphlib::Shape::create(shape_vec));
            continue;
        }

        auto new_shapes = shape_range<N>(min_shape, max_shape, idx + 1);
        shapes.reserve(shapes.size() + new_shapes.size());
        shapes.insert(shapes.end(), new_shapes.begin(), new_shapes.end());
    }
    return shapes;
}

inline bool valid_inputs(const graphlib::Shape& in1, const graphlib::Shape& in2)
{
    return in1 == in2 || graphlib::can_be_broadcasted(in1, in2);
}

inline std::vector<std::vector<graphlib::Shape>> generate_input_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range(std::array<uint32_t, 1>{1}, std::array<uint32_t, 1>{16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range(std::array<uint32_t, 2>{1, 1}, std::array<uint32_t, 2>{16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range(std::array<uint32_t, 3>{1, 1, 1}, std::array<uint32_t, 3>{5, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range(std::array<uint32_t, 4>{1, 1, 1, 1}, std::array<uint32_t, 4>{5, 1, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        input_shapes.push_back({shapes[i], shapes[i]});
        for (size_t j = i + 1; j < shapes.size(); ++j)
        {
            if (valid_inputs(shapes[i], shapes[j]))
            {
                input_shapes.push_back({shapes[i], shapes[j]});
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

UNARY_ELTWISE_SWEEP_TEST_SET(
    UnaryEltwiseOpsSweep,
    testing::Values(tt::ops::OpType::Abs, tt::ops::OpType::Sine, tt::ops::OpType::Cosine),
    SimpleOpTest);

}  // namespace tt::test::ops::eltwise_unary
