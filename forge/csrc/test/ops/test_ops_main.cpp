// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/embed.h>
#pragma clang diagnostic pop

#include "test_ops.hpp"

namespace tt::test::ops
{

// Tests the decomposition of the operation by running the decomposition pass on the initial graph and verifying that
// the evaluation of the decomposed graph matches the evalution of the initial graph.
TEST_P(SimpleOpTest, test_decompose)
{
    // Run the decomposition pass on the initial graph.
    run_decompose_graph();
    // Evaluate the graph node by node.
    auto outputs = eval_graph();
    compare_with_golden(outputs);
}

// Tests backward implementation for the operation by running the autograd engine on the
// initial graph and verifying that the evaluation of the backward graph matches the torch `output.backward()` call.
TEST_P(SimpleOpTest, test_backward)
{
    pybind11::gil_scoped_release gil_release;  // release GIL for PyTorch autograd
    run_autograd();

    // Confirm that the forward pass produced the expected output.
    auto outputs = eval_graph();
    compare_with_golden(outputs);

    auto computed_grads = eval_graph(graphlib::NodeEpochType::Backward);
    verify_bwd_gradients(computed_grads);
}

// Only test decompose for ops without backward support
TEST_P(SimpleOpDecomposeOnlyTest, test_decompose)
{
    // Run the decomposition pass on the initial graph.
    run_decompose_graph();
    // Evaluate the graph node by node.
    auto outputs = eval_graph();
    compare_with_golden(outputs);
}

}  // namespace tt::test::ops

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Set up the global test environment with flags parsed from command line arguments.
    // This allows for configuration of test behavior, such as enabling dumping graphs to reportify.
    GraphTestFlags flags = GraphTestFlags::from_cli_args(argc, argv);
    ::testing::AddGlobalTestEnvironment(new GraphTestFlagsEnvironment(flags));

    pybind11::scoped_interpreter guard{};

    // Import torch module - needed in cases where we interact with torch python but before the
    // forge module is imported.
    pybind11::module::import("torch");

    return RUN_ALL_TESTS();
}
