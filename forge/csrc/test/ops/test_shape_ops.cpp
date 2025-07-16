// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gtest/internal/gtest-param-util.h>
#include <torch/torch.h>

#include <array>
#include <initializer_list>
#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/python_bindings.hpp"
#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "passes/decomposing_context.hpp"
#include "pybind11/gil.h"
#include "test/common.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::shape_ops
{

// Helper function to create Reshape ops with shape attribute
tt::ops::Op create_reshape_op(const std::vector<int>& target_shape)
{
    tt::ops::Op op(tt::ops::OpType::Reshape);
    op.set_attr("shape", target_shape);
    return op;
}

// Helper function to create Transpose ops with dim attributes
tt::ops::Op create_transpose_op(int dim0, int dim1)
{
    tt::ops::Op op(tt::ops::OpType::Transpose);
    op.set_attr("dim0", dim0);
    op.set_attr("dim1", dim1);
    return op;
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeOp,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Values(
            std::make_tuple(create_reshape_op({1, 1, 1, 24}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 1, 6, 4}), VecShapes{{2, 12}}),
            std::make_tuple(create_reshape_op({1, 1, 24, 1}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 1, 24, 1}), VecShapes{{24, 1}}),
            std::make_tuple(create_reshape_op({1, 2, 2, 6}), VecShapes{{2, 2, 2, 3}}),
            std::make_tuple(create_reshape_op({1, 2, 3, 4}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 24, 1, 1}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 4, 3, 2}), VecShapes{{1, 2, 3, 4}}),

            std::make_tuple(create_reshape_op({2, 3, 4}), VecShapes{{1, 2, 3, 4}}),

            std::make_tuple(create_reshape_op({1, 24}), VecShapes{{2, 12}}),
            std::make_tuple(create_reshape_op({6, 4}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({8, 3}), VecShapes{{24, 1}}),
            std::make_tuple(create_reshape_op({24, 1}), VecShapes{{24, 1}}),

            std::make_tuple(create_reshape_op({24}), VecShapes{{1, 2, 3, 4}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

INSTANTIATE_TEST_SUITE_P(
    TransposeOp,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Values(
            std::make_tuple(create_transpose_op(0, 1), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(0, 2), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(0, 3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(1, 2), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(1, 3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(2, 3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(-1, -2), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(-2, -3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(-3, -4), VecShapes{{1, 2, 3, 4}}),

            std::make_tuple(create_transpose_op(0, 1), VecShapes{{5, 4, 3}}),
            std::make_tuple(create_transpose_op(0, 2), VecShapes{{5, 4, 3}}),
            std::make_tuple(create_transpose_op(1, 2), VecShapes{{5, 4, 3}}),
            std::make_tuple(create_transpose_op(-1, -2), VecShapes{{5, 4, 3}}),

            std::make_tuple(create_transpose_op(0, 1), VecShapes{{3, 4}}),
            std::make_tuple(create_transpose_op(-1, -2), VecShapes{{3, 4}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

// Range tests for Reshape
INSTANTIATE_TEST_SUITE_P(
    ReshapeRangeOp,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(
                create_reshape_op({1, 1, 1, 20}),
                create_reshape_op({1, 1, 20, 1}),
                create_reshape_op({1, 20, 1, 1}),
                create_reshape_op({20, 1, 1, 1}),
                create_reshape_op({2, 10}),
                create_reshape_op({4, 5}),
                create_reshape_op({5, 4}),
                create_reshape_op({20})),
            testing::Values(VecShapes{{4, 5}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

// Note: STANDARD_RANGE_OP_TEST_SET generates shapes with varying dimensions,
// but transpose dims must be valid for the generated shape dimensions

}  // namespace tt::test::ops::shape_ops
