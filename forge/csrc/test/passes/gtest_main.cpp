// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "test/common.hpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Set up the global test environment with flags parsed from command line arguments.
    // This allows for configuration of test behavior, such as enabling dumping graphs to reportify.
    GraphTestFlags flags = GraphTestFlags::from_cli_args(argc, argv);
    ::testing::AddGlobalTestEnvironment(new GraphTestFlagsEnvironment(flags));

    pybind11::scoped_interpreter guard{};
    return RUN_ALL_TESTS();
}
