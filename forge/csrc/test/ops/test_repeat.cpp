// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "test_ops.hpp"

namespace tt::test::ops::repeat
{

tt::ops::Op create_repeat_op(const std::vector<int>& repeats)
{
    return tt::ops::Op(tt::ops::OpType::Repeat, {{"repeats", repeats}});
}

struct RepeatTestCase
{
    std::vector<graphlib::Shape> input_shapes;
    std::vector<int> repeats;
    std::string description;
};

std::vector<RepeatTestCase> get_repeat_test_cases()
{
    return {
        // Basic functionality tests - same number of dimensions
        {{{2}}, {1}, "1D no repeat"},
        {{{2}}, {3}, "1D simple repeat"},
        {{{1, 2}}, {1, 1}, "2D no repeat"},
        {{{1, 2}}, {2, 1}, "2D repeat first dim"},
        {{{1, 2}}, {1, 3}, "2D repeat second dim"},
        {{{2, 3}}, {2, 1}, "2D repeat first dim only"},

        // Dimension expansion tests (triggers decompose_initial)
        {{{2}}, {1, 2}, "1D to 2D expansion"},
        {{{2}}, {2, 1}, "1D to 2D expansion with repeat"},
        {{{1, 2}}, {1, 1, 1}, "2D to 3D expansion"},
        {{{1, 2}}, {2, 1, 1}, "2D to 3D expansion with repeat"},
        {{{3}}, {2, 1, 3}, "1D to 3D expansion"},
        {{{2, 4}}, {1, 3, 2, 1}, "2D to 4D expansion"},

        // Edge cases for shape function
        {{{1}}, {1}, "Single element no repeat"},
        {{{1}}, {5}, "Single element with repeat"},
        {{{1, 1}}, {1, 1}, "All ones no repeat"},
        {{{1, 1}}, {3, 2}, "All ones with repeat"},

    };
}

std::vector<OpTestParam> generate_repeat_params()
{
    std::vector<OpTestParam> params;

    for (const auto& test_case : get_repeat_test_cases())
    {
        tt::ops::Op op = create_repeat_op(test_case.repeats);
        params.emplace_back(op, test_case.input_shapes);
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    RepeatOp,
    SimpleOpDecomposeOnlyTest,  // Repeat has no backward implementation
    testing::ValuesIn(generate_repeat_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::repeat
