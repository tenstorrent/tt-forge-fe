// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <functional>

#include "nlohmann/json_fwd.hpp"
#include "reportify/paths.hpp"

namespace tt {

namespace graphlib {
    class Graph;
    class Node;
}

namespace reportify
{
using json = nlohmann::json;

void dump_graph(
    const std::string& path,
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    const std::string& report_path = get_pass_reports_relative_directory());

// Default path
json create_json_for_graph(
    const graphlib::Graph *graph,
    std::function<bool(graphlib::Node*)> node_filter = [](graphlib::Node*) { return true; });

void dump_graph(
    const std::string& test_name,
    const std::string& graph_prefix,
    const graphlib::Graph* graph,
    const std::string& report_path = get_pass_reports_relative_directory());

void dump_consteval_graph(const std::string& test_name, const std::string& graph_prefix, const graphlib::Graph* graph);

void dump_epoch_type_graphs(
        const std::string& test_name,
        const std::string& graph_prefix,
        const graphlib::Graph *graph,
        const std::string& directory_path = get_default_reportify_path(""));

void dump_epoch_id_graphs(
        const std::string& test_name,
        const std::string& graph_prefix,
        const graphlib::Graph *graph,
        const std::string& directory_path = get_default_reportify_path(""));
}  // namespace reportify

} // tt
