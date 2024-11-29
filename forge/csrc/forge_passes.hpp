// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "passes/dataformat.hpp"
#include "passes/fracture.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace tt
{

using NodeId = graphlib::NodeId;
using PortId = graphlib::PortId;
void lower_reshape(graphlib::Graph *, graphlib::OpNode *node);  // unused

// Run post initial graph passes
std::tuple<std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>>, passes::FractureChipIdAssignments>
run_post_initial_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object, passes::FractureGroups const &fracture_groups);
void run_optimization_graph_passes(graphlib::Graph *graph);
std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_optimize_decompose_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object);
std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_autograd_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object);

// Pre-lowering passes, last-minute changes before going to MLIR
graphlib::Graph *run_pre_lowering_passes(
    graphlib::Graph *graph, const std::optional<DataFormat> default_df_override = {});
}  // namespace tt
