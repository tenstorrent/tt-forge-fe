// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::layernorm
{

tt::ops::Op create_layernorm_op(int dim, float epsilon)
{
    return tt::ops::Op(tt::ops::OpType::Layernorm, {{"dim", dim}, {"epsilon", epsilon}});
}

std::vector<VecShapes> get_layernorm_individual_test_shapes()
{
    return {
        // Basic 2D shapes
        VecShapes{{2, 8}, {1, 8}, {1, 8}},     // [input, gamma, beta]
        VecShapes{{4, 16}, {1, 16}, {1, 16}},  // Larger embedding dim
        VecShapes{{1, 32}, {1, 32}, {1, 32}},  // Single batch

        // 3D shapes
        VecShapes{{2, 4, 8}, {1, 1, 8}, {1, 1, 8}},     // Batch size 2
        VecShapes{{1, 8, 16}, {1, 1, 16}, {1, 1, 16}},  // Single batch, seq length 8
        VecShapes{{3, 6, 32}, {1, 1, 32}, {1, 1, 32}},  // Larger dimensions

        // Edge cases
        VecShapes{{1, 1}, {1, 1}, {1, 1}},     // Minimal shape
        VecShapes{{8, 64}, {1, 64}, {1, 64}},  // Larger batch
    };
}

std::vector<OpTestParam> generate_layernorm_sweep_params()
{
    std::vector<OpTestParam> params;

    std::vector<float> epsilons = {1e-5f, 1e-6f, 1e-4f};

    // Various input shapes (batch_size, ..., feature_dim)
    std::vector<VecShapes> sweep_shapes = {
        // 2D shapes
        VecShapes{{1, 16}, {1, 16}, {1, 16}},
        VecShapes{{2, 32}, {1, 32}, {1, 32}},
        VecShapes{{4, 64}, {1, 64}, {1, 64}},

        // 3D shapes
        VecShapes{{1, 4, 16}, {1, 1, 16}, {1, 1, 16}},
        VecShapes{{2, 8, 32}, {1, 1, 32}, {1, 1, 32}},
        VecShapes{{1, 16, 64}, {1, 1, 64}, {1, 1, 64}},
    };

    for (float epsilon : epsilons)
    {
        for (const auto& shapes : sweep_shapes)
        {
            // Test with dim = -1 (last dimension)
            params.emplace_back(create_layernorm_op(-1, epsilon), shapes);
        }
    }

    return params;
}

std::vector<OpTestParam> generate_layernorm_dimension_tests()
{
    std::vector<OpTestParam> params;

    // Test both dim=-1 and dim=(last_dim_index) for various rank tensors
    std::vector<std::pair<VecShapes, std::vector<int>>> dim_test_cases = {
        // {shapes, {valid_dims}}
        {VecShapes{{4, 8}, {1, 8}, {1, 8}}, {-1, 1}},              // 2D: dim=-1 or dim=1
        {VecShapes{{2, 4, 16}, {1, 1, 16}, {1, 1, 16}}, {-1, 2}},  // 3D: dim=-1 or dim=2
    };

    float epsilon = 1e-5f;

    for (const auto& [shapes, dims] : dim_test_cases)
    {
        for (int dim : dims)
        {
            params.emplace_back(create_layernorm_op(dim, epsilon), shapes);
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    LayerNormIndividual,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(create_layernorm_op(-1, 1e-5f)), testing::ValuesIn(get_layernorm_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    LayerNormSweep,
    SimpleOpTest,
    testing::ValuesIn(generate_layernorm_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    LayerNormDimensions,
    SimpleOpTest,
    testing::ValuesIn(generate_layernorm_dimension_tests()),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::layernorm
