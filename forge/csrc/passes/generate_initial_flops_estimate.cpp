// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/generate_initial_flops_estimate.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

void generate_initial_flops_estimate(graphlib::Graph *graph)
{
    long total_flops = 0;
    int total_ops = 0;
    std::map<std::string, std::tuple<long, int>> macs_per_op;
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
        if (not op)
            continue;

        std::vector<std::vector<std::uint32_t>> operand_tuples;
        for (auto data_operand : graph->data_operands(node))
            operand_tuples.push_back(data_operand->shape().as_vector());

        long flops = op->op().initial_flops_estimate(operand_tuples);

        if (macs_per_op.find(op->op().as_string()) == macs_per_op.end())
        {
            macs_per_op[op->op().as_string()] = std::make_pair(flops, 1);
        }
        else
        {
            macs_per_op[op->op().as_string()] = std::make_pair(
                std::get<0>(macs_per_op[op->op().as_string()]) + flops,
                std::get<1>(macs_per_op[op->op().as_string()]) + 1);
        }
        total_flops += flops;
        total_ops += 1;
    }

    log_trace(LogGraphCompiler, "Initial FLOPs Estimate:");
    log_trace(LogGraphCompiler, "OP, COUNT, FLOPs");
    for (const auto &p : macs_per_op)
    {
        log_trace(LogGraphCompiler, "{}, {}, {}", p.first, std::get<1>(p.second), std::get<0>(p.second));
    }
    if (env_as<bool>("FORGE_SHOW_FLOPS_ESTIMATE"))
        log_info(
            LogGraphCompiler, "Initial flops estimate from Forge: {}B, total_ops: {}", total_flops / 1e9, total_ops);
    else
        log_trace(
            LogGraphCompiler, "Initial flops estimate from Forge: {}B, total_ops: {}", total_flops / 1e9, total_ops);
}
}  // namespace tt::passes
