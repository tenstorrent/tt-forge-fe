// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "autograd/autograd.hpp"
#include "graph_lib/utils.hpp"
#include "passes/decomposing_context.hpp"
#include "test_ops.hpp"

namespace tt::test::ops
{
TEST_P(SimpleOpTest, test_decompose)
{
    // TODO: decomposing context needs `compiler_cfg`; passing nullptr for now...
    tt::decompose_tt_forge_graph<DecomposeEpoch::Initial>(get_graph(), std::shared_ptr<void>(nullptr, [](void *) {}));

    // Evaluate the graph node by node.
    auto outputs = eval_graph();
    compare_with_golden(outputs);
}

// Tests backward implementation for the operation by running the autograd engine on the
// initial graph and verifying that the evaluation of the backward graph matches the torch `output.backward()` call.
TEST_P(SimpleOpTest, test_backward)
{
    run_autograd();

    // Confirm that the forward pass produced the expected output.
    auto outputs = eval_graph();
    compare_with_golden(outputs);

    auto computed_grads = eval_graph(graphlib::NodeEpochType::Backward);
    verify_bwd_gradients(computed_grads);
}
}  // namespace tt::test::ops

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    pybind11::scoped_interpreter guard{};
    return RUN_ALL_TESTS();
}
