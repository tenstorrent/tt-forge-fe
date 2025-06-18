// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include "graph_lib/defines.hpp"
#include "lower_to_forge/common.hpp"
#include "passes/amp.hpp"
#include "python_bindings_common.hpp"

namespace tt
{
using DfMap = std::unordered_map<std::string, DataFormat>;
using MfMap = std::unordered_map<std::string, MathFidelity>;
using AMPNodeProperties = passes::AMPNodeProperties;

struct DeviceConfig;
}  // namespace tt

namespace tt::graphlib
{
class Graph;
class Node;
}  // namespace tt::graphlib

namespace tt::passes
{
void satisfy_data_format_constraints(graphlib::Graph *graph, bool fp32_acc_supported);
void validate_data_formats(const graphlib::Graph *graph, const DeviceConfig &device_config);
void configure_output_data_formats(graphlib::Graph *graph, std::optional<DataFormat> default_df_override);
void apply_user_data_format_override(graphlib::Graph *graph, py::object compiler_cfg_object);

void run_dataformat_passes(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    const std::optional<DataFormat> default_df_override,
    const std::optional<DataFormat> default_accumulate_df,
    const DataFormat fp32_fallback,
    const MathFidelity default_math_fidelity,
    const int amp_level,
    const std::vector<AMPNodeProperties> &amp_properties);

}  // namespace tt::passes
