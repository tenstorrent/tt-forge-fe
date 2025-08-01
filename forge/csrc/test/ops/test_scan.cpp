// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::scan
{
std::vector<VecShapes> get_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 1, 32}},
        VecShapes{{1, 1, 32, 1}},
        VecShapes{{1, 32, 1, 1}},
        VecShapes{{32, 1, 1, 1}},
        VecShapes{{1, 2, 3, 4}},
        VecShapes{{2, 3, 4}},
        VecShapes{{3, 4}},
        VecShapes{{4}},
        VecShapes{{1}}};
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
        input_shapes.push_back({shapes[i]});
    }

    return input_shapes;
}

/**
 * Testing cumulative sum operation.
 */
const std::vector<OpTestParam> get_cumsum_test_params(const std::vector<VecShapes>& in_shapes)
{
    std::vector<OpTestParam> params;

    for (const auto& shape_vec : in_shapes)
    {
        for (const auto& shape : shape_vec)
        {
            for (int dim = 0; dim < static_cast<int>(shape.size()); ++dim)
            {
                tt::ops::Op op1(tt::ops::OpType::CumulativeSum, {{"dim", static_cast<int>(dim)}});
                tt::ops::Op op2(tt::ops::OpType::CumulativeSum, {{"dim", -static_cast<int>(dim) - 1}});
                params.emplace_back(op1, std::vector{shape});
                params.emplace_back(op2, std::vector{shape});
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    CumulativeSumOpNoBackwardIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_cumsum_test_params(get_individual_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    CumulativeSumOpNoBackwardSweeps,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_cumsum_test_params(generate_input_shapes())),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::scan
