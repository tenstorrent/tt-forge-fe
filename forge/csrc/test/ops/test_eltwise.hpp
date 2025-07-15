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

namespace tt::test::ops::eltwise_binary
{

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

STANDARD_SWEEP_OP_TEST_SET(UnaryEltwiseOpsSweep, tt::ops::OpType::Abs, SimpleOpTest);

}  // namespace tt::test::ops::eltwise_unary
