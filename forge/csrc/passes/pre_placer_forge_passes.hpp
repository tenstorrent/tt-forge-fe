// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backend_api/device_config.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_forge/common.hpp"
#include "passes/fracture.hpp"

namespace tt
{

using Graph = graphlib::Graph;
using Node = graphlib::Node;
using NodeType = graphlib::NodeType;
using NodeContext = graphlib::NodeContext;
using NodeToNodeMap = std::unordered_map<Node *, Node *>;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;

using DfMap = std::unordered_map<std::string, DataFormat>;
using MfMap = std::unordered_map<std::string, MathFidelity>;

void lower_to_buffering_queues(Graph *graph);

void tag_subgraph_nodes(graphlib::Graph *graph);

void split_unsupported_gradient_ops(graphlib::Graph *graph, const DeviceConfig &device_config);
void remove_nops(graphlib::Graph *graph);
void insert_queues_for_op_intermediates(graphlib::Graph *graph, const std::vector<std::string> &op_intermediates);

void fix_transposes(graphlib::Graph *graph, const DeviceConfig &device_config);

void fix_tms_on_output(graphlib::Graph *graph);

void sanitize_past_cache_ios(graphlib::Graph *graph);

void fix_untilized_outputs(graphlib::Graph *graph, const DeviceConfig &device_config);

void fix_host_inputs(graphlib::Graph *graph);

void replace_buffers_with_nops(graphlib::Graph *graph);

void insert_nop_on_matmul_input(graphlib::Graph *graph);

void insert_tilize_op_on_input(graphlib::Graph *graph);

std::vector<std::vector<std::string>> update_epoch_breaks_for_partial_datacopy(
    graphlib::Graph *graph, std::vector<std::vector<std::string>> const &op_names_to_epoch_break);

void calculate_ublock_order(graphlib::Graph *graph);

void apply_user_override_data_formats(
    graphlib::Graph *graph, std::optional<DataFormat> default_df_override, DataFormat default_accumulate_df);

void insert_partial_datacopy_tms(graphlib::Graph *graph);
void split_broadcasts(Graph *graph);
void constant_pre_broadcast(Graph *graph);
std::tuple<Node *, Edge, Edge> split_broadcast(Graph *graph, Node *consumer, Edge &edge, int idx_of_broadcast_tm);

void validate_buffering_queues(graphlib::Graph *graph);

void lower_fallback_data_formats(graphlib::Graph *graph, DataFormat fp32_fallback, bool fp32_acc_supported);

// Convert TTForge graph to Forge graph
std::unique_ptr<Graph> lower_to_forge_ops(Graph *graph);

void apply_math_fidelity(graphlib::Graph *graph, const MathFidelity default_math_fidelity);

void fix_data_formats(graphlib::Graph *graph, bool fix_data_formats);

void validate_data_formats(graphlib::Graph *graph);

bool is_relu_in_forge_attrs(const ForgeOpAttrs &forge_attrs);

bool has_hoistable_relu(graphlib::Graph *graph, Node *node);

void insert_user_defined_queues(
    graphlib::Graph *graph, const std::vector<std::tuple<std::string, std::string, int>> &insert_queues);
}  // namespace tt
