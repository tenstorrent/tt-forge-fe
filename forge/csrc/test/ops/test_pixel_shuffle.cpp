// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::pixel_shuffle
{
/**
 * Create PixelShuffle operation with specified attributes
 */
tt::ops::Op create_pixel_shuffle_op(int upscale_factor)
{
    return tt::ops::Op(tt::ops::OpType::PixelShuffle, {{"upscale_factor", upscale_factor}});
}

std::vector<VecShapes> get_pixel_shuffle_individual_test_shapes()
{
    return {
        // Basic 4D pixel shuffle tests - (N, C*r*r, H, W)
        VecShapes{{1, 4, 8, 8}},   // upscale_factor=2: 1*4, 8*8 -> 1*1, 16*16
        VecShapes{{1, 9, 4, 4}},   // upscale_factor=3: 1*9, 4*4 -> 1*1, 12*12
        VecShapes{{1, 16, 2, 2}},  // upscale_factor=4: 1*16, 2*2 -> 1*1, 8*8

        // Different batch sizes
        VecShapes{{2, 4, 6, 6}},  // upscale_factor=2: 2*4, 6*6 -> 2*1, 12*12
        VecShapes{{3, 9, 3, 3}},  // upscale_factor=3: 3*9, 3*3 -> 3*1, 9*9

        // Different channel counts
        VecShapes{{1, 8, 5, 5}},   // upscale_factor=2: 1*8, 5*5 -> 1*2, 10*10
        VecShapes{{1, 12, 4, 4}},  // upscale_factor=2: 1*12, 4*4 -> 1*3, 8*8
        VecShapes{{1, 36, 2, 2}},  // upscale_factor=3: 1*36, 2*2 -> 1*4, 6*6

        // 5D pixel shuffle tests (batch + extra dim)
        VecShapes{{2, 3, 4, 6, 6}},  // upscale_factor=2: 2*3*4, 6*6 -> 2*3*1, 12*12
        VecShapes{{1, 2, 9, 3, 3}},  // upscale_factor=3: 1*2*9, 3*3 -> 1*2*1, 9*9

        // Edge case: minimal sizes
        VecShapes{{1, 4, 1, 1}},  // upscale_factor=2: 1*4, 1*1 -> 1*1, 2*2
        VecShapes{{1, 9, 1, 1}},  // upscale_factor=3: 1*9, 1*1 -> 1*1, 3*3
    };
}

std::vector<std::vector<graphlib::Shape>> get_pixel_shuffle_sweeps_test_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    // 4D sweeps with upscale_factor=2
    for (uint32_t n : {1, 2})
    {
        for (uint32_t c : {1, 2, 3})
        {  // c*4 channels total
            for (uint32_t h : {2, 4, 8})
            {
                for (uint32_t w : {2, 4, 8})
                {
                    input_shapes.push_back({{n, c * 4, h, w}});  // upscale_factor=2 needs c*4 channels
                }
            }
        }
    }

    // 4D sweeps with upscale_factor=3
    for (uint32_t n : {1, 2})
    {
        for (uint32_t c : {1, 2})
        {  // c*9 channels total
            for (uint32_t h : {2, 3})
            {
                for (uint32_t w : {2, 3})
                {
                    input_shapes.push_back({{n, c * 9, h, w}});  // upscale_factor=3 needs c*9 channels
                }
            }
        }
    }

    // 5D sweeps
    for (uint32_t batch : {1, 2})
    {
        for (uint32_t extra : {1, 2})
        {
            input_shapes.push_back({{batch, extra, 4, 3, 3}});  // upscale_factor=2
            input_shapes.push_back({{batch, extra, 9, 2, 2}});  // upscale_factor=3
        }
    }

    return input_shapes;
}

/**
 * Testing pixel shuffle operation.
 */
const std::vector<OpTestParam> get_pixel_shuffle_test_params(const std::vector<VecShapes>& in_shapes)
{
    std::vector<OpTestParam> params;

    for (const auto& shape_vec : in_shapes)
    {
        for (const auto& shape : shape_vec)
        {
            auto shape_vec_copy = shape.as_vector();
            uint32_t channel_dim = shape_vec_copy[shape_vec_copy.size() - 3];

            // Test different upscale factors that divide the channel dimension
            for (int upscale_factor : {2, 3, 4})
            {
                if (channel_dim % (upscale_factor * upscale_factor) == 0)
                    params.emplace_back(
                        tt::ops::Op(tt::ops::OpType::PixelShuffle, {{"upscale_factor", upscale_factor}}),
                        std::vector{shape});
            }
        }
    }

    return params;
}

/**
 * Individual test suite - comprehensive test cases with backward pass
 */
INSTANTIATE_TEST_SUITE_P(
    PixelShuffleOpIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_pixel_shuffle_test_params(get_pixel_shuffle_individual_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

/**
 * Sweep test suite - systematic coverage with decompose only
 */
INSTANTIATE_TEST_SUITE_P(
    PixelShuffleOpSweeps,
    SimpleOpDecomposeOnlyTest,
    testing::ValuesIn(get_pixel_shuffle_test_params(get_pixel_shuffle_sweeps_test_shapes())),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::pixel_shuffle
