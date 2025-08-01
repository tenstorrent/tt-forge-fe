// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::training
{

static std::vector<tt::ops::OpType> training_ops = {
    tt::ops::OpType::Dropout,
};

std::vector<VecShapes> get_training_ops_test_shapes()
{
    return {
        VecShapes{{1}},
        VecShapes{{4}},
        VecShapes{{3, 4}},
        VecShapes{{2, 3, 4}},
        VecShapes{{1, 2, 3, 4}},
        VecShapes{{1, 1, 1, 32}},
        VecShapes{{1, 1, 32, 1}},
        VecShapes{{1, 32, 1, 1}},
        VecShapes{{32, 1, 1, 1}},
    };
}

/**
 * Generates various test params for testing dropout op.
 * Note: we need to always generate the same seed in order to test backward pass, because new op is generated and if
 * seed is different, we will get different results in verification.
 */
const std::vector<OpTestParam> generate_dropout_op_params()
{
    const auto shapes = get_training_ops_test_shapes();

    std::vector<OpTestParam> params;
    params.reserve(shapes.size() * 2);

    for (const auto& shape : shapes)
    {
        for (const float f : {0.0f, 0.1f, 0.2f, 0.5f, 0.7f, 1.0f})
        {
            for (const bool training : {true, false})
            {
                tt::ops::Op op(tt::ops::OpType::Dropout, {{"p", f}, {"training", training}, {"seed", 0}});
                params.push_back({op, std::vector{shape}});
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    DropoutOpIndividualTest,
    SimpleOpTest,
    testing::ValuesIn(generate_dropout_op_params()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::training
