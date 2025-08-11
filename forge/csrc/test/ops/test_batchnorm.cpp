// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::batchnorm
{

/// Creates batchnorm ops with different epsilon values
std::vector<tt::ops::Op> get_batchnorm_ops()
{
    std::vector<tt::ops::Op> ops;
    for (float epsilon : {1e-5f, 1e-4f, 1e-3f, 1e-2f, 0.1f})
    {
        ops.push_back(tt::ops::Op(tt::ops::OpType::Batchnorm, {{"epsilon", epsilon}}));
    }
    return ops;
}

/// Batchnorm requires 5 inputs: input, weight, bias, running_mean, running_var
/// Weight, bias, running_mean, running_var should have shape [C] where C is the number of channels
std::vector<VecShapes> get_batchnorm_individual_test_shapes()
{
    return {
        // [input, weight, bias, running_mean, running_var]
        // 2D cases (NLP scenarios)
        VecShapes{{1, 512}, {512}, {512}, {512}, {512}},        // Single sequence
        VecShapes{{8, 768}, {768}, {768}, {768}, {768}},        // Batch of sequences
        VecShapes{{16, 1024}, {1024}, {1024}, {1024}, {1024}},  // Large hidden size

        // 4D cases (CNN scenarios)
        VecShapes{{1, 32, 14, 14}, {32}, {32}, {32}, {32}},     // Basic CNN
        VecShapes{{2, 64, 7, 7}, {64}, {64}, {64}, {64}},       // Batch size 2
        VecShapes{{1, 128, 3, 3}, {128}, {128}, {128}, {128}},  // Small spatial
        VecShapes{{4, 16, 28, 28}, {16}, {16}, {16}, {16}},     // Larger batch
        VecShapes{{1, 256, 1, 1}, {256}, {256}, {256}, {256}},  // Global pooled
        VecShapes{{8, 32, 32, 32}, {32}, {32}, {32}, {32}},     // Square spatial
    };
}

/// Validates that the input shapes are compatible for batchnorm
bool valid_inputs(const std::vector<graphlib::Shape>& shapes)
{
    if (shapes.size() != 5)
        return false;

    const auto& input_shape = shapes[0];
    const auto& weight_shape = shapes[1];
    const auto& bias_shape = shapes[2];
    const auto& running_mean_shape = shapes[3];
    const auto& running_var_shape = shapes[4];

    // Input must have at least 2 dimensions (N, C, ...)
    if (input_shape.size() < 2)
        return false;

    // Get channel dimension (second dimension)
    uint32_t channels = input_shape.as_vector()[1];

    // Weight, bias, running_mean, running_var must all be 1D with size = channels
    return weight_shape.size() == 1 && weight_shape.as_vector()[0] == channels && bias_shape.size() == 1 &&
           bias_shape.as_vector()[0] == channels && running_mean_shape.size() == 1 &&
           running_mean_shape.as_vector()[0] == channels && running_var_shape.size() == 1 &&
           running_var_shape.as_vector()[0] == channels;
}

std::vector<std::vector<graphlib::Shape>> generate_input_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    // Different batch sizes
    std::vector<uint32_t> batch_sizes = {1, 2, 4, 8};
    // Different channel sizes
    std::vector<uint32_t> channel_sizes = {16, 32, 64, 128};
    // Different spatial dimensions (2D and 4D only)
    std::vector<std::vector<uint32_t>> spatial_dims = {
        {},      // 2D: (N, C) - NLP scenarios
        {7, 7},  // 4D: (N, C, H, W) - CNN scenarios
    };

    for (uint32_t batch_size : batch_sizes)
    {
        for (uint32_t channels : channel_sizes)
        {
            for (const auto& spatial : spatial_dims)
            {
                std::vector<uint32_t> input_shape = {batch_size, channels};
                input_shape.insert(input_shape.end(), spatial.begin(), spatial.end());

                std::vector<graphlib::Shape> shapes = {
                    graphlib::Shape::create(input_shape),
                    graphlib::Shape::create({channels}),
                    graphlib::Shape::create({channels}),
                    graphlib::Shape::create({channels}),
                    graphlib::Shape::create({channels})};

                if (valid_inputs(shapes))
                {
                    input_shapes.push_back(shapes);
                }
            }
        }
    }

    return input_shapes;
}

INSTANTIATE_TEST_SUITE_P(
    BatchnormIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_batchnorm_ops()), testing::ValuesIn(get_batchnorm_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    BatchnormSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_batchnorm_ops()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    BatchnormDefault,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(tt::ops::Op(tt::ops::OpType::Batchnorm, {{"epsilon", 1e-5f}})),
            testing::ValuesIn(get_batchnorm_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::batchnorm
