// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::embedding
{

tt::ops::Op create_embedding_op() { return tt::ops::Op(tt::ops::OpType::Embedding); }

tt::ops::Op create_embedding_bw_op() { return tt::ops::Op(tt::ops::OpType::EmbeddingBw); }

std::vector<VecShapes> get_embedding_individual_test_shapes()
{
    return {
        // 1D indices cases: [indices_shape, weights_shape]
        VecShapes{{1}, {5, 8}},    // Single index
        VecShapes{{4}, {10, 32}},  // Multiple 1D indices
        VecShapes{{8}, {20, 16}},  // Larger 1D indices

        // 2D indices cases
        VecShapes{{1, 1}, {10, 32}},    // 2D indices [1,1]
        VecShapes{{2, 3}, {20, 64}},    // 2D indices [2,3]
        VecShapes{{1, 5}, {15, 128}},   // 2D indices [1,5]
        VecShapes{{4, 2}, {25, 256}},   // 2D indices [4,2]
        VecShapes{{3, 3}, {100, 512}},  // Larger 2D indices
    };
}

std::vector<VecShapes> get_embedding_bw_individual_test_shapes()
{
    return {
        // 1D indices: [indices_shape, weights_shape, grad_shape]
        VecShapes{{1}, {5, 8}, {1, 8}},     // Single index
        VecShapes{{4}, {10, 32}, {4, 32}},  // Multiple 1D indices
        VecShapes{{8}, {20, 16}, {8, 16}},  // Larger 1D indices

        // 2D indices
        VecShapes{{1, 1}, {10, 32}, {1, 1, 32}},    // 2D indices [1,1]
        VecShapes{{2, 3}, {20, 64}, {2, 3, 64}},    // 2D indices [2,3]
        VecShapes{{1, 5}, {15, 128}, {1, 5, 128}},  // 2D indices [1,5]
        VecShapes{{4, 2}, {25, 256}, {4, 2, 256}},  // 2D indices [4,2]
    };
}

std::vector<OpTestParam> generate_valid_embedding_individual_params()
{
    std::vector<OpTestParam> params;

    // Curated individual test cases - all guaranteed to be valid, no validation needed
    auto test_shapes = get_embedding_individual_test_shapes();

    for (const auto& shapes : test_shapes)
    {
        params.emplace_back(create_embedding_op(), shapes);
    }

    return params;
}

std::vector<OpTestParam> generate_valid_embedding_sweep_params()
{
    std::vector<OpTestParam> params;

    // Generate combinations of 1D and 2D indices with 2D embedding tables
    std::vector<uint32_t> vocab_sizes = {5, 10, 20, 50};
    std::vector<uint32_t> emb_dims = {4, 8, 16, 32};

    // Valid index shapes: 1D and 2D only
    std::vector<std::vector<uint32_t>> index_shapes = {
        {1},
        {2},
        {4},
        {8},  // 1D indices
        {1, 1},
        {1, 2},
        {2, 1},
        {2, 2},
        {3, 2},
        {2, 3},
        {4, 2}  // 2D indices
    };

    for (auto vocab_size : vocab_sizes)
    {
        for (auto emb_dim : emb_dims)
        {
            for (const auto& idx_shape : index_shapes)
            {
                std::vector<graphlib::Shape> shapes = {
                    graphlib::Shape::create(idx_shape), graphlib::Shape::create({vocab_size, emb_dim})};
                params.emplace_back(create_embedding_op(), shapes);
            }
        }
    }

    return params;
}

std::vector<OpTestParam> generate_valid_embedding_bw_individual_params()
{
    std::vector<OpTestParam> params;

    // Curated individual test cases - all guaranteed to be valid, no validation needed
    auto test_shapes = get_embedding_bw_individual_test_shapes();

    for (const auto& shapes : test_shapes)
    {
        params.emplace_back(create_embedding_bw_op(), shapes);
    }

    return params;
}

std::vector<OpTestParam> generate_valid_embedding_bw_sweep_params()
{
    std::vector<OpTestParam> params;

    // Generate based on embedding sweep parameters, constructing gradient shapes
    auto embedding_params = generate_valid_embedding_sweep_params();

    for (const auto& embedding_param : embedding_params)
    {
        const auto& indices_shape = embedding_param.input_shapes[0];
        const auto& weights_shape = embedding_param.input_shapes[1];

        // Create gradient shape: indices_shape + [emb_dim]
        std::vector<uint32_t> grad_shape_vec;
        for (size_t i = 0; i < indices_shape.size(); ++i)
        {
            grad_shape_vec.push_back(indices_shape[i]);
        }
        grad_shape_vec.push_back(weights_shape[1]);  // embedding_dim

        std::vector<graphlib::Shape> shapes = {indices_shape, weights_shape, graphlib::Shape::create(grad_shape_vec)};

        params.emplace_back(create_embedding_bw_op(), shapes);
    }

    return params;
}

// Embedding tests with backward pass - pre-filtered valid parameters only
INSTANTIATE_TEST_SUITE_P(
    EmbeddingIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_valid_embedding_individual_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    EmbeddingSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_valid_embedding_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

// Embedding_BW tests without backward pass using custom test class
INSTANTIATE_TEST_SUITE_P(
    EmbeddingBwIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_valid_embedding_bw_individual_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    EmbeddingBwSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_valid_embedding_bw_sweep_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::embedding
