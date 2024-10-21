// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{
void extract_unique_op_configuration(
    graphlib::Graph* graph,
    std::string stage,
    const std::optional<std::vector<std::string>>& supported_ops = std::nullopt);
}
