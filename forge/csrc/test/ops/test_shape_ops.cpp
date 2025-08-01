// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::shape_ops
{

tt::ops::Op create_reshape_op(const std::vector<int>& target_shape)
{
    tt::ops::Op op(tt::ops::OpType::Reshape);
    op.set_attr("shape", target_shape);
    return op;
}

tt::ops::Op create_transpose_op(int dim0, int dim1)
{
    tt::ops::Op op(tt::ops::OpType::Transpose);
    op.set_attr("dim0", dim0);
    op.set_attr("dim1", dim1);
    return op;
}

tt::ops::Op create_squeeze_op(int dim)
{
    tt::ops::Op op(tt::ops::OpType::Squeeze);
    op.set_attr("dim", dim);
    return op;
}

tt::ops::Op create_unsqueeze_op(int dim)
{
    tt::ops::Op op(tt::ops::OpType::Unsqueeze);
    op.set_attr("dim", dim);
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

INSTANTIATE_TEST_SUITE_P(
    SqueezeOp,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Values(
            // Squeeze dimension 0
            std::make_tuple(create_squeeze_op(0), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_squeeze_op(0), VecShapes{{1, 1, 1, 1}}),
            std::make_tuple(create_squeeze_op(0), VecShapes{{1, 32}}),
            // Squeeze dimension 1
            std::make_tuple(create_squeeze_op(1), VecShapes{{2, 1, 3, 4}}),
            std::make_tuple(create_squeeze_op(1), VecShapes{{32, 1}}),
            // Squeeze dimension 2
            std::make_tuple(create_squeeze_op(2), VecShapes{{1, 2, 1, 4}}),
            std::make_tuple(create_squeeze_op(2), VecShapes{{2, 3, 1}}),
            // Squeeze dimension 3
            std::make_tuple(create_squeeze_op(3), VecShapes{{1, 2, 3, 1}}),
            // Squeeze with negative indices
            std::make_tuple(create_squeeze_op(-1), VecShapes{{1, 2, 3, 1}}),
            std::make_tuple(create_squeeze_op(-2), VecShapes{{1, 2, 1, 4}}),
            std::make_tuple(create_squeeze_op(-3), VecShapes{{1, 1, 3, 4}}),
            std::make_tuple(create_squeeze_op(-4), VecShapes{{1, 2, 3, 4}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

INSTANTIATE_TEST_SUITE_P(
    UnsqueezeOp,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Values(
            // Unsqueeze at dimension 0
            std::make_tuple(create_unsqueeze_op(0), VecShapes{{2, 3, 4}}),
            std::make_tuple(create_unsqueeze_op(0), VecShapes{{32}}),
            std::make_tuple(create_unsqueeze_op(0), VecShapes{{3, 4}}),
            // Unsqueeze at dimension 1
            std::make_tuple(create_unsqueeze_op(1), VecShapes{{2, 3, 4}}),
            std::make_tuple(create_unsqueeze_op(1), VecShapes{{32}}),
            // Unsqueeze at dimension 2
            std::make_tuple(create_unsqueeze_op(2), VecShapes{{2, 3, 4}}),
            std::make_tuple(create_unsqueeze_op(2), VecShapes{{3, 4}}),
            // Unsqueeze at dimension 3
            std::make_tuple(create_unsqueeze_op(3), VecShapes{{2, 3, 4}}),
            // Unsqueeze at end
            std::make_tuple(create_unsqueeze_op(3), VecShapes{{2, 3, 4}}),
            std::make_tuple(create_unsqueeze_op(2), VecShapes{{3, 4}}),
            std::make_tuple(create_unsqueeze_op(1), VecShapes{{32}}),
            // Unsqueeze with negative indices
            std::make_tuple(create_unsqueeze_op(-1), VecShapes{{2, 3, 4}}),
            std::make_tuple(create_unsqueeze_op(-2), VecShapes{{2, 3, 4}}),
            std::make_tuple(create_unsqueeze_op(-3), VecShapes{{2, 3, 4}}),
            std::make_tuple(create_unsqueeze_op(-1), VecShapes{{3, 4}}),
            std::make_tuple(create_unsqueeze_op(-1), VecShapes{{32}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

}  // namespace tt::test::ops::shape_ops
