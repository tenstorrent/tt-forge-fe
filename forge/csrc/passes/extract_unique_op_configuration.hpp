// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "graph_lib/node_types.hpp"

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{

using OpShapesType = std::vector<graphlib::Shape>;
using OpAttrsType = std::vector<graphlib::OpType>;
using OpShapesAttrsType = std::pair<OpShapesType, OpAttrsType>;
using UniqueOpShapesAttrsType = std::map<std::string, std::vector<OpShapesAttrsType>>;

UniqueOpShapesAttrsType extract_unique_op_configuration(
    graphlib::Graph* graph, const std::optional<std::vector<std::string>>& supported_ops = std::nullopt);

void print_unique_op_configuration(const UniqueOpShapesAttrsType& unique_op_shapes_attrs, std::string op_config_info);

void export_unique_op_configuration_to_csv_file(
    const UniqueOpShapesAttrsType& unique_op_shapes_attrs, std::string graph_name, std::string stage);

void extract_unique_op_configuration(
    graphlib::Graph* graph,
    std::string stage,
    const std::optional<std::vector<std::string>>& supported_ops = std::nullopt);
}  // namespace tt::passes
