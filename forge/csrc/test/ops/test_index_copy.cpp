// SPDX-FileCopyrightText: � 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"
#include "torch/torch.h"

namespace tt::test::ops::index_copy
{

bool valid_inputs(const std::vector<graphlib::Shape>& shapes, int dim)
{
    const graphlib::Shape& input_shape = shapes[0];
    const graphlib::Shape& index_shape = shapes[1];
    const graphlib::Shape& src_shape = shapes[2];

    int original_dim = dim;
    if (dim < 0)
    {
        dim += input_shape.size();
    }

    // Index tensor must be 1D
    if (index_shape.size() != 1)
    {
        return false;
    }

    // Check if index values don't exceed the dimension size
    uint32_t dim_size = input_shape[dim];
    uint32_t num_indices = index_shape[0];

    // For simplicity, assume indices are valid (0 to dim_size-1)
    // In real tests, we'd need to check actual index values
    if (num_indices > dim_size)
    {
        return false;
    }

    // Special handling for KV cache decomposition path
    // When dim == -2 and we have 4D tensors, it gets decomposed to KV cache operations
    // which require index tensor's first dimension to match input's first dimension
    if ((original_dim == 2 || original_dim == -2) && input_shape.size() == 4 && src_shape.size() == 4)
    {
        // For KV cache path, index tensor needs to match batch dimension
        if (num_indices != input_shape[0])
        {
            return false;
        }
    }

    // Source tensor must have compatible shape for insertion
    // For the two supported cases:
    // 1. Inserting from 0 to i without gaps: src should match input except at dim
    // 2. Inserting on some index but only one slice: src should have size 1 at dim
    bool case1 = (src_shape[dim] == num_indices);            // Sequential insertion
    bool case2 = (src_shape[dim] == 1 && num_indices == 1);  // Single slice insertion

    if (!case1 && !case2)
    {
        return false;
    }

    // Check that other dimensions match
    for (size_t i = 0; i < input_shape.size(); ++i)
    {
        if (i == static_cast<size_t>(dim))
        {
            continue;  // Already checked above
        }
        if (src_shape[i] != input_shape[i])
        {
            return false;
        }
    }

    return true;
}

std::vector<OpTestParam> generate_index_copy_test_params()
{
    // Input tensor shapes - covering supported cases
    auto input_shapes = {
        graphlib::Shape{8},           // 1D: [8]
        graphlib::Shape{4, 8},        // 2D: [4, 8]
        graphlib::Shape{2, 4, 8},     // 3D: [2, 4, 8]
        graphlib::Shape{1, 3, 4, 8},  // 4D: [1, 3, 4, 8] - Use batch=1 for KV cache testing
    };

    std::vector<OpTestParam> params;

    // Generate test cases for each input shape
    for (const auto& input_shape : input_shapes)
    {
        // Test each dimension
        for (int dim = 0; dim < static_cast<int>(input_shape.size()); ++dim)
        {
            uint32_t dim_size = input_shape[dim];

            // Case 1: Sequential insertion from 0 to i (without gaps)
            for (uint32_t num_indices = 1; num_indices <= std::min(dim_size, 3u); ++num_indices)
            {
                // Special case for KV cache path (4D tensor with dim=2)
                if (dim == 2 && input_shape.size() == 4)
                {
                    // For KV cache, index tensor needs to match batch dimension (first dimension)
                    graphlib::Shape index_shape{input_shape[0]};
                    graphlib::Shape src_shape = input_shape;
                    src_shape[dim] = num_indices;

                    std::vector<graphlib::Shape> shapes = {input_shape, index_shape, src_shape};
                    std::vector<torch::Dtype> dtypes = {
                        torch::kFloat, torch::kLong, torch::kFloat};  // input, index, source
                    if (valid_inputs(shapes, dim))
                    {
                        // Create sequential index tensor from 0 to num_indices-1
                        torch::Tensor sequential_indices =
                            torch::arange(0, static_cast<int64_t>(index_shape[0]), torch::kLong);
                        std::vector<torch::Tensor> override_tensors = {
                            {}, sequential_indices, {}};  // Only override index tensor

                        tt::ops::Op op(tt::ops::OpType::IndexCopy, {{"dim", dim}});
                        params.emplace_back(op, shapes, dtypes, override_tensors);
                    }
                }
                else
                {
                    graphlib::Shape index_shape{num_indices};
                    graphlib::Shape src_shape = input_shape;
                    src_shape[dim] = num_indices;  // Source has same size as indices

                    std::vector<graphlib::Shape> shapes = {input_shape, index_shape, src_shape};
                    std::vector<torch::Dtype> dtypes = {
                        torch::kFloat, torch::kLong, torch::kFloat};  // input, index, source
                    if (valid_inputs(shapes, dim))
                    {
                        // Create sequential index tensor from 0 to num_indices-1
                        torch::Tensor sequential_indices =
                            torch::arange(0, static_cast<int64_t>(index_shape[0]), torch::kLong);
                        std::vector<torch::Tensor> override_tensors = {
                            {}, sequential_indices, {}};  // Only override index tensor

                        tt::ops::Op op(tt::ops::OpType::IndexCopy, {{"dim", dim}});
                        params.emplace_back(op, shapes, dtypes, override_tensors);
                    }
                }
            }

            // Case 2: Single slice insertion at multiple specific indices
            if (dim_size > 0)
            {
                // Special case for KV cache path (4D tensor with dim=2)
                if (dim == 2 && input_shape.size() == 4)
                {
                    // For KV cache, index tensor needs to match batch dimension (first dimension)
                    graphlib::Shape index_shape{input_shape[0]};
                    graphlib::Shape src_shape = input_shape;
                    src_shape[dim] = 1;  // Single slice

                    std::vector<graphlib::Shape> shapes = {input_shape, index_shape, src_shape};
                    std::vector<torch::Dtype> dtypes = {
                        torch::kFloat, torch::kLong, torch::kFloat};  // input, index, source
                    if (valid_inputs(shapes, dim))
                    {
                        // Create sequential index tensor from 0 to num_indices-1
                        torch::Tensor sequential_indices =
                            torch::arange(0, static_cast<int64_t>(index_shape[0]), torch::kLong);
                        std::vector<torch::Tensor> override_tensors = {
                            {}, sequential_indices, {}};  // Only override index tensor

                        tt::ops::Op op(tt::ops::OpType::IndexCopy, {{"dim", dim}});
                        params.emplace_back(op, shapes, dtypes, override_tensors);
                    }
                }
                else
                {
                    // Generate multiple test cases for single slice insertion at different indices
                    uint32_t num_test_indices = std::min(dim_size, 3u);  // Test up to 3 different indices
                    for (uint32_t test_idx = 0; test_idx < num_test_indices; ++test_idx)
                    {
                        graphlib::Shape index_shape{1};  // Single index
                        graphlib::Shape src_shape = input_shape;
                        src_shape[dim] = 1;  // Single slice

                        std::vector<graphlib::Shape> shapes = {input_shape, index_shape, src_shape};
                        std::vector<torch::Dtype> dtypes = {
                            torch::kFloat, torch::kLong, torch::kFloat};  // input, index, source
                        if (valid_inputs(shapes, dim))
                        {
                            // Create random index tensor within valid range for single slice insertion
                            int64_t max_index = static_cast<int64_t>(dim_size - 1);
                            int64_t random_index = std::rand() % (max_index + 1);  // Random index from 0 to max_index
                            torch::Tensor random_index_tensor = torch::tensor({random_index}, torch::kLong);
                            std::vector<torch::Tensor> override_tensors = {
                                {}, random_index_tensor, {}};  // Only override index tensor

                            tt::ops::Op op(tt::ops::OpType::IndexCopy, {{"dim", dim}});
                            params.emplace_back(op, shapes, dtypes, override_tensors);
                        }
                    }
                }
            }
        }

        // Test negative dimensions
        for (int neg_dim = -1; neg_dim >= -static_cast<int>(input_shape.size()); --neg_dim)
        {
            int dim = neg_dim + input_shape.size();
            uint32_t dim_size = input_shape[dim];

            // Only test single slice insertion for negative dims to keep tests manageable
            if (dim_size > 0)
            {
                // Special case for KV cache path (4D tensor with neg_dim=-2, which is dim=2)
                if (neg_dim == -2 && input_shape.size() == 4)
                {
                    // For KV cache, index tensor needs to match batch dimension (first dimension)
                    graphlib::Shape index_shape{input_shape[0]};
                    graphlib::Shape src_shape = input_shape;
                    src_shape[dim] = 1;

                    std::vector<graphlib::Shape> shapes = {input_shape, index_shape, src_shape};
                    std::vector<torch::Dtype> dtypes = {
                        torch::kFloat, torch::kLong, torch::kFloat};  // input, index, source
                    if (valid_inputs(shapes, neg_dim))
                    {
                        // For KV cache, use sequential indices starting from 0
                        torch::Tensor sequential_indices =
                            torch::arange(0, static_cast<int64_t>(input_shape[0]), torch::kLong);
                        std::vector<torch::Tensor> override_tensors = {
                            {}, sequential_indices, {}};  // Only override index tensor

                        tt::ops::Op op(tt::ops::OpType::IndexCopy, {{"dim", neg_dim}});
                        params.emplace_back(op, shapes, dtypes, override_tensors);
                    }
                }
                else
                {
                    graphlib::Shape index_shape{1};
                    graphlib::Shape src_shape = input_shape;
                    src_shape[dim] = 1;

                    std::vector<graphlib::Shape> shapes = {input_shape, index_shape, src_shape};
                    std::vector<torch::Dtype> dtypes = {
                        torch::kFloat, torch::kLong, torch::kFloat};  // input, index, source
                    if (valid_inputs(shapes, neg_dim))
                    {
                        // Create random index tensor within valid range for single slice insertion
                        int64_t max_index = static_cast<int64_t>(dim_size - 1);
                        int64_t random_index = std::rand() % (max_index + 1);  // Random index from 0 to max_index
                        torch::Tensor random_index_tensor = torch::tensor({random_index}, torch::kLong);
                        std::vector<torch::Tensor> override_tensors = {
                            {}, random_index_tensor, {}};  // Only override index tensor

                        tt::ops::Op op(tt::ops::OpType::IndexCopy, {{"dim", neg_dim}});
                        params.emplace_back(op, shapes, dtypes, override_tensors);
                    }
                }
            }
        }
    }

    return params;
}

// Test individual cases using the standard SimpleOpDecomposeOnlyTest
INSTANTIATE_TEST_SUITE_P(
    IndexCopyOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(generate_index_copy_test_params()),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::index_copy
