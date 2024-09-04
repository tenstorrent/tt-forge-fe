// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "passes/dataformat.hpp"
#include "passes/fracture.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace tt {

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

// Run lowering passes
std::unique_ptr<graphlib::Graph> run_pre_placer_forge_passes(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    std::vector<std::uint32_t> chip_ids = {0},
    const std::vector<std::string> &op_names_dont_fuse = {},
    const std::vector<std::string> &op_names_manual_fuse = {},
    const passes::FractureChipIdAssignments &fracture_chip_id_assignments = {},
    const std::optional<DataFormat> default_df_override = {},
    const std::optional<DataFormat> default_accumulate_df = {},
    const bool enable_broadcast_splitting = false,
    const DataFormat fp32_fallback = DataFormat::Float16_b,
    const MathFidelity default_math_fidelity = MathFidelity::HiFi3,
    const bool enable_auto_fusing = false,
    const int amp_level = 0,
    const bool enable_recompute = false,
    const bool output_queues_on_host = true,
    const bool input_queues_on_host = true,
    const std::vector<std::tuple<std::string, std::string, int>> &insert_queues = {},
    std::vector<AMPNodeProperties> amp_properties = {},
    const std::vector<std::string> &op_intermediates_to_save = {},
    bool use_interactive_placer = true,
    bool enable_device_tilize = false);

// Pre-lowering passes, last-minute changes before going to MLIR
graphlib::Graph* run_pre_lowering_passes(
    graphlib::Graph *graph,
    const std::optional<DataFormat> default_df_override = {});
}
