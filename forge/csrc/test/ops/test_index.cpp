// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "test_ops.hpp"

namespace tt::test::ops::index
{

tt::ops::Op create_index_op(int dim, int start, int stop, int stride)
{
    return tt::ops::Op(tt::ops::OpType::Index, {{"dim", dim}, {"start", start}, {"stop", stop}, {"stride", stride}});
}

std::vector<OpTestParam> generate_index_focused_params()
{
    return {// Basic positive dim - covers eval() basic path and shape() positive dim conversion
            {create_index_op(0, 0, 4, 1), {graphlib::Shape{8}}},

            // Negative dim - covers shape() and backward() negative dim normalization
            {create_index_op(-1, 0, 4, 1), {graphlib::Shape{4, 8}}},

            // Negative start - covers shape() negative start handling (lines 63-66)
            {create_index_op(0, -2, 8, 1), {graphlib::Shape{8}}},

            // Single element - covers backward() single iteration case
            {create_index_op(0, 3, 4, 1), {graphlib::Shape{8}}},

            // Multi-dim positive - covers shape() multi-dimensional positive dim
            {create_index_op(1, 0, 2, 1), {graphlib::Shape{2, 4, 8}}},

            // Multi-dim negative - covers shape() and backward() multi-dim negative handling
            {create_index_op(-2, 1, 3, 1), {graphlib::Shape{2, 4, 8}}},

            // Boundary case - covers shape() bounds check (new_size >= 0)
            {create_index_op(0, 0, 1, 1), {graphlib::Shape{4}}},

            // Middle slice - covers general slicing behavior
            {create_index_op(0, 2, 6, 1), {graphlib::Shape{8}}},

            // Last element with negative index - combines negative start with boundary
            {create_index_op(0, -1, 8, 1), {graphlib::Shape{8}}},

            // 4D tensor - covers higher-dimensional tensor handling
            {create_index_op(2, 1, 3, 1), {graphlib::Shape{2, 3, 4, 5}}},

            // Negative stop - covers backward() negative stop normalization
            {create_index_op(0, 2, -1, 1), {graphlib::Shape{8}}}};
}

INSTANTIATE_TEST_SUITE_P(
    IndexFocused,
    SimpleOpTest,
    testing::ValuesIn(generate_index_focused_params()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::index
