// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "test/common.hpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    GraphTestFlags flags = GraphTestFlags::from_cli_args(argc, argv);

    // Set up the test environment with the specified flags.
    ::testing::AddGlobalTestEnvironment(new GraphTestFlagsEnvironment(flags));

    pybind11::scoped_interpreter guard{};
    return RUN_ALL_TESTS();
}
