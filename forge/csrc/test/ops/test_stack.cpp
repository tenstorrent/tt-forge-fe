// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::stack
{

tt::ops::Op create_stack_op(int dim) { return tt::ops::Op(tt::ops::OpType::Stack, {{"dim", dim}}); }

std::vector<OpTestParam> generate_stack_test_params()
{
    std::vector<OpTestParam> params;

    // 1D tensors - test basic positive/negative dims to cover (dim < 0) branch
    std::vector<graphlib::Shape> shapes_1d = {graphlib::Shape{4}, graphlib::Shape{4}};
    params.emplace_back(create_stack_op(0), shapes_1d);   // Positive: stack at beginning
    params.emplace_back(create_stack_op(1), shapes_1d);   // Positive: stack at end
    params.emplace_back(create_stack_op(-1), shapes_1d);  // Negative: stack at end (tests dim < 0 branch)
    params.emplace_back(create_stack_op(-2), shapes_1d);  // Negative: stack at beginning (tests dim < 0 branch)

    // 2D tensors - test middle insertion and negative indexing
    std::vector<graphlib::Shape> shapes_2d = {graphlib::Shape{2, 4}, graphlib::Shape{2, 4}};
    params.emplace_back(create_stack_op(1), shapes_2d);   // Positive: stack in middle
    params.emplace_back(create_stack_op(-1), shapes_2d);  // Negative: stack at end (tests dim < 0 branch)
    params.emplace_back(create_stack_op(-3), shapes_2d);  // Negative: stack at beginning (tests dim < 0 branch)

    // 3D tensors - test more complex negative indexing
    std::vector<graphlib::Shape> shapes_3d = {graphlib::Shape{2, 3, 4}, graphlib::Shape{2, 3, 4}};
    params.emplace_back(create_stack_op(0), shapes_3d);   // Positive: stack at beginning
    params.emplace_back(create_stack_op(-1), shapes_3d);  // Negative: stack at end (tests dim < 0 branch)
    params.emplace_back(create_stack_op(-4), shapes_3d);  // Negative: stack at beginning (tests dim < 0 branch)

    // 3 inputs - test multiple tensor stacking
    std::vector<graphlib::Shape> shapes_3_inputs = {graphlib::Shape{3}, graphlib::Shape{3}, graphlib::Shape{3}};
    params.emplace_back(create_stack_op(0), shapes_3_inputs);   // Positive
    params.emplace_back(create_stack_op(-1), shapes_3_inputs);  // Negative (tests dim < 0 branch)

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    StackOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_stack_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

std::vector<OpTestParam> generate_stack_sweep_params()
{
    std::vector<OpTestParam> sweep_params;

    // Select a few key shapes for sweep testing
    auto key_shapes = {
        graphlib::Shape{2},        // 1D
        graphlib::Shape{2, 3},     // 2D
        graphlib::Shape{2, 2, 2},  // 3D
    };

    for (const auto& shape : key_shapes)
    {
        // Test with 2 inputs
        std::vector<graphlib::Shape> shapes = {shape, shape};

        // Test first, middle, and last positions (positive and negative)
        sweep_params.emplace_back(create_stack_op(0), shapes);   // First position
        sweep_params.emplace_back(create_stack_op(-1), shapes);  // Last position (negative, tests dim < 0 branch)

        if (shape.size() > 1)
        {
            int mid = static_cast<int>(shape.size()) / 2;
            sweep_params.emplace_back(create_stack_op(mid), shapes);  // Middle position
            sweep_params.emplace_back(
                create_stack_op(-mid - 1), shapes);  // Middle position (negative, tests dim < 0 branch)
        }
    }

    return sweep_params;
}

INSTANTIATE_TEST_SUITE_P(
    StackOpsSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_stack_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    StackOpsEdgeCases,
    SimpleOpDecomposeOnlyTest,
    testing::Values(
        // Single element tensors
        OpTestParam{create_stack_op(0), {graphlib::Shape{1}, graphlib::Shape{1}}},
        OpTestParam{create_stack_op(-1), {graphlib::Shape{1}, graphlib::Shape{1}}},  // Tests dim < 0 branch

        // Multiple inputs (4 tensors)
        OpTestParam{
            create_stack_op(0), {graphlib::Shape{2}, graphlib::Shape{2}, graphlib::Shape{2}, graphlib::Shape{2}}},
        OpTestParam{
            create_stack_op(-1),
            {// Tests dim < 0 branch with multiple inputs
             graphlib::Shape{2},
             graphlib::Shape{2},
             graphlib::Shape{2},
             graphlib::Shape{2}}},

        // 4D tensors - test extreme negative indexing (tests dim < 0 branch)
        OpTestParam{create_stack_op(-1), {graphlib::Shape{1, 2, 3, 4}, graphlib::Shape{1, 2, 3, 4}}},  // Append
        OpTestParam{create_stack_op(-5), {graphlib::Shape{1, 2, 3, 4}, graphlib::Shape{1, 2, 3, 4}}}   // Prepend
        ),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::stack
