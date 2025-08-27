// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::pad
{

std::vector<OpTestParam> generate_pad_test_params()
{
    std::vector<OpTestParam> params;

    std::vector<std::tuple<graphlib::Shape, std::vector<int>, int, float, bool>> test_cases = {
        // Basic 3D cases - channel_last=false
        {graphlib::Shape{1, 3, 8}, {1, 1}, 0, 0.0f, false},  // constant mode, basic padding
        {graphlib::Shape{1, 3, 8}, {1, 1}, 0, 2.5f, false},  // constant mode, different value
        {graphlib::Shape{1, 3, 8}, {1, 1}, 1, 0.0f, false},  // replicate mode
        {graphlib::Shape{1, 3, 8}, {1, 1}, 2, 0.0f, false},  // reflect mode

        {graphlib::Shape{1, 3, 8}, {0, 2}, 0, 0.0f, false},  // constant mode, right-only
        {graphlib::Shape{1, 3, 8}, {0, 2}, 1, 0.0f, false},  // replicate mode, right-only
        {graphlib::Shape{1, 3, 8}, {0, 2}, 2, 0.0f, false},  // reflect mode, right-only

        {graphlib::Shape{1, 3, 8}, {2, 0}, 0, 0.0f, false},  // constant mode, left-only
        {graphlib::Shape{1, 3, 8}, {2, 0}, 1, 0.0f, false},  // replicate mode, left-only
        {graphlib::Shape{1, 3, 8}, {2, 0}, 2, 0.0f, false},  // reflect mode, left-only

        // 4D cases with 2D padding - channel_last=false only
        {graphlib::Shape{1, 3, 8, 12}, {1, 1, 1, 1}, 0, 0.0f, false},  // constant mode, symmetric
        {graphlib::Shape{1, 3, 8, 12}, {1, 1, 1, 1}, 1, 0.0f, false},  // replicate mode, symmetric
        {graphlib::Shape{1, 3, 8, 12}, {1, 1, 1, 1}, 2, 0.0f, false},  // reflect mode, symmetric

        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 1, 1}, 0, 0.0f, false},  // constant mode, height-only
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 1, 1}, 1, 0.0f, false},  // replicate mode, height-only
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 1, 1}, 2, 0.0f, false},  // reflect mode, height-only

        {graphlib::Shape{1, 3, 8, 12}, {1, 1, 0, 0}, 0, 0.0f, false},  // constant mode, width-only
        {graphlib::Shape{1, 3, 8, 12}, {1, 1, 0, 0}, 1, 0.0f, false},  // replicate mode, width-only
        {graphlib::Shape{1, 3, 8, 12}, {1, 1, 0, 0}, 2, 0.0f, false},  // reflect mode, width-only

        // Zero padding (should become nop)
        {graphlib::Shape{1, 3, 8}, {0, 0}, 0, 0.0f, false},  // constant mode, no padding
        {graphlib::Shape{1, 3, 8}, {0, 0}, 1, 0.0f, false},  // replicate mode, no padding
        {graphlib::Shape{1, 3, 8}, {0, 0}, 2, 0.0f, false},  // reflect mode, no padding

        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 0, 0}, 0, 0.0f, false},  // constant mode, no 2D padding
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 0, 0}, 1, 0.0f, false},  // replicate mode, no 2D padding
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 0, 0}, 2, 0.0f, false},  // reflect mode, no 2D padding

        // Channel-last coverage - test all 3 modes
        // NOTE: PyTorch has internal validation that rejects certain 4D tensor shapes for replicate/reflect modes
        // after channel_last permutation. For example, {1, 8, 12, 3} -> {1, 3, 8, 12} fails with:
        // "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported" even though 4D should work.
        {graphlib::Shape{1, 8, 12, 3}, {1, 1}, 0, 0.0f, true},  // 4D NHWC, constant mode (works fine)
        {graphlib::Shape{8, 12, 3}, {1, 1}, 1, 0.0f, true},     // 3D HWC, replicate mode (avoid PyTorch restriction)
        {graphlib::Shape{8, 12, 3}, {1, 1}, 2, 0.0f, true},     // 3D HWC, reflect mode (avoid PyTorch restriction)
    };

    // Create OpTestParam objects from test cases
    for (const auto& [shape, padding, mode, value, channel_last] : test_cases)
    {
        tt::ops::Op op(
            tt::ops::OpType::Pad,
            {{"padding", padding}, {"mode", mode}, {"value", value}, {"channel_last", channel_last}});
        params.emplace_back(op, std::vector{shape});
    }

    return params;
}

std::vector<OpTestParam> generate_edge_case_params()
{
    std::vector<OpTestParam> params;

    std::vector<std::tuple<graphlib::Shape, std::vector<int>, int, float, bool>> edge_cases = {
        // Single-sided padding tests
        {graphlib::Shape{1, 3, 8, 12}, {1, 0, 0, 0}, 0, 0.0f, false},  // Left only
        {graphlib::Shape{1, 3, 8, 12}, {0, 1, 0, 0}, 0, 0.0f, false},  // Right only
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 1, 0}, 0, 0.0f, false},  // Top only
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 0, 1}, 0, 0.0f, false},  // Bottom only

        {graphlib::Shape{1, 3, 8, 12}, {1, 0, 0, 0}, 1, 0.0f, false},  // Left only, replicate
        {graphlib::Shape{1, 3, 8, 12}, {0, 1, 0, 0}, 1, 0.0f, false},  // Right only, replicate
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 1, 0}, 1, 0.0f, false},  // Top only, replicate
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 0, 1}, 1, 0.0f, false},  // Bottom only, replicate

        {graphlib::Shape{1, 3, 8, 12}, {1, 0, 0, 0}, 2, 0.0f, false},  // Left only, reflect
        {graphlib::Shape{1, 3, 8, 12}, {0, 1, 0, 0}, 2, 0.0f, false},  // Right only, reflect
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 1, 0}, 2, 0.0f, false},  // Top only, reflect
        {graphlib::Shape{1, 3, 8, 12}, {0, 0, 0, 1}, 2, 0.0f, false},  // Bottom only, reflect

        // Different tensor sizes
        {graphlib::Shape{2, 5, 10}, {1, 1}, 0, 0.0f, false},  // Different 3D shape
        {graphlib::Shape{2, 5, 10}, {1, 1}, 1, 0.0f, false},  // Different 3D shape, replicate
        {graphlib::Shape{2, 5, 10}, {1, 1}, 2, 0.0f, false},  // Different 3D shape, reflect
    };

    for (const auto& [shape, padding, mode, value, channel_last] : edge_cases)
    {
        tt::ops::Op op(
            tt::ops::OpType::Pad,
            {{"padding", padding}, {"mode", mode}, {"value", value}, {"channel_last", channel_last}});
        params.emplace_back(op, std::vector{shape});
    }

    return params;
}

// Specific test cases for boundary conditions in reflect/replicate modes
std::vector<OpTestParam> generate_boundary_test_params()
{
    std::vector<OpTestParam> params;

    // Simple boundary test cases
    std::vector<std::tuple<graphlib::Shape, std::vector<int>, int, float, bool>> boundary_cases = {
        // Safe small tensor tests
        {graphlib::Shape{1, 4, 6}, {1, 1}, 1, 0.0f, false},  // Small tensor, replicate
        {graphlib::Shape{1, 4, 6}, {1, 1}, 2, 0.0f, false},  // Small tensor, reflect (1 < 6, safe)
        {graphlib::Shape{1, 4, 6}, {2, 1}, 1, 0.0f, false},  // Small tensor, replicate, asymmetric
        {graphlib::Shape{1, 4, 6}, {2, 1}, 2, 0.0f, false},  // Small tensor, reflect, asymmetric (2 < 6, safe)

        {graphlib::Shape{2, 3, 5}, {1, 1}, 1, 0.0f, false},  // Minimal tensor, replicate
        {graphlib::Shape{2, 3, 5}, {1, 1}, 2, 0.0f, false},  // Minimal tensor, reflect (1 < 5, safe)
        {graphlib::Shape{2, 3, 5}, {1, 2}, 1, 0.0f, false},  // Minimal tensor, replicate, asymmetric
        {graphlib::Shape{2, 3, 5}, {1, 2}, 2, 0.0f, false},  // Minimal tensor, reflect, asymmetric (2 < 5, safe)
    };

    // Create OpTestParam objects from boundary cases
    for (const auto& [shape, padding, mode, value, channel_last] : boundary_cases)
    {
        tt::ops::Op op(
            tt::ops::OpType::Pad,
            {{"padding", padding}, {"mode", mode}, {"value", value}, {"channel_last", channel_last}});
        params.emplace_back(op, std::vector{shape});
    }

    return params;
}

// Instantiate the main test suite with comprehensive parameters (decompose only - no backward support yet)
INSTANTIATE_TEST_SUITE_P(
    PadOpsComprehensive,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_pad_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Instantiate edge case tests (decompose only - no backward support yet)
INSTANTIATE_TEST_SUITE_P(
    PadOpsEdgeCases,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_edge_case_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Instantiate boundary condition tests (decompose only - no backward support yet)
INSTANTIATE_TEST_SUITE_P(
    PadOpsBoundary,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_boundary_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Test cases that demonstrate PyTorch dimension restrictions - these should fail
// These should throw an exception due to PyTorch's dimension restrictions
// We expect a c10::Error with a message about padding dimension support
// NOTE: PyTorch has internal validation that rejects certain 4D tensor shapes for replicate/reflect modes
// after channel_last permutation. For example, {1, 8, 12, 3} -> {1, 3, 8, 12} fails with:
// "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported" even though 4D should work.
std::vector<OpTestParam> generate_pytorch_failure_test_params()
{
    std::vector<OpTestParam> params;

    // These specific cases fail due to PyTorch's internal validation for replicate/reflect modes
    // after channel_last permutation transforms 4D tensor shapes
    std::vector<std::tuple<graphlib::Shape, std::vector<int>, int, float, bool>> failure_cases = {
        // Original failing cases from our investigation - demonstrate PyTorch dimension restrictions
        {graphlib::Shape{1, 8, 12, 3}, {1, 1}, 1, 0.0f, true},  // 4D tensor, replicate mode, channel_last=true
        {graphlib::Shape{1, 8, 12, 3}, {1, 1}, 2, 0.0f, true},  // 4D tensor, reflect mode, channel_last=true
    };

    for (const auto& [shape, padding, mode, value, channel_last] : failure_cases)
    {
        tt::ops::Op op(
            tt::ops::OpType::Pad,
            {{"padding", padding}, {"mode", mode}, {"value", value}, {"channel_last", channel_last}});
        params.emplace_back(op, std::vector{shape});
    }

    return params;
}

class PadPyTorchFailureTest : public ::testing::TestWithParam<OpTestParam>
{
};

TEST_P(PadPyTorchFailureTest, ExpectedPyTorchDimensionFailures)
{
    auto param = GetParam();
    tt::ops::Op op = param.op;
    std::vector<graphlib::Shape> shapes = param.input_shapes;

    // Create input tensor
    at::Tensor input = torch::randn({(long)shapes[0][0], (long)shapes[0][1], (long)shapes[0][2], (long)shapes[0][3]});
    std::vector<at::Tensor> inputs = {input};

    EXPECT_THROW({ op.eval(inputs); }, c10::Error);
}

INSTANTIATE_TEST_SUITE_P(
    PadOpsPyTorchDimensionFailures,
    PadPyTorchFailureTest,
    testing::ValuesIn(generate_pytorch_failure_test_params()),
    [](const testing::TestParamInfo<PadPyTorchFailureTest::ParamType>& info)
    {
        const auto& op = info.param.op;
        const auto& shapes = info.param.input_shapes;

        auto padding = op.attr_as<std::vector<int>>("padding");
        auto mode = op.attr_as<int>("mode");

        std::string mode_str = (mode == 1) ? "replicate" : "reflect";

        return "pytorch_fail_" + std::to_string(shapes[0][0]) + "x" + std::to_string(shapes[0][1]) + "x" +
               std::to_string(shapes[0][2]) + "x" + std::to_string(shapes[0][3]) + "_" + mode_str + "_channel_last";
    });

}  // namespace tt::test::ops::pad
