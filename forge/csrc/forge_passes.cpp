// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "forge_passes.hpp"

#include <algorithm>
#include <map>

#include "backend_api/device_config.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/query.hpp"
#include "graph_lib/utils.hpp"
#include "passes/constant_folding.hpp"
#include "passes/dataformat.hpp"
#include "passes/decomposing_context.hpp"
#include "passes/erase_consecutive_reshape.hpp"
#include "passes/erase_inverse_ops.hpp"
#include "passes/erase_unnecessary_4d_tm_sequence.hpp"
#include "passes/mlir_compiler.hpp"
#include "passes/explicate_unsqueeze.hpp"
#include "passes/fuse_conv2d_bias.hpp"
#include "passes/fuse_pad_conv2d.hpp"
#include "passes/fuse_per_channel_ops.hpp"
#include "passes/fuse_redundant_tm_sequence.hpp"
#include "passes/generate_initial_flops_estimate.hpp"
#include "passes/hoist_transforms_to_inputs.hpp"
#include "passes/insert_inverse_on_io.hpp"
#include "passes/limit_to_4d_reshape.hpp"
#include "passes/link_past_cache_ios.hpp"
#include "passes/lower_concat_to_runtime_transform.hpp"
#include "passes/lower_reinterpret_shape.hpp"
#include "passes/lowering_context.hpp"
#include "passes/move_requantize.hpp"
#include "passes/move_select_after_matmul_optional.hpp"
#include "passes/pad_output_buffer.hpp"
#include "passes/passes_utils.hpp"
#include "passes/post_autograd_graph_passes.hpp"
#include "passes/pre_lowering_passes.hpp"
#include "passes/pre_placer_forge_passes.hpp"
#include "passes/print_graph.hpp"
#include "passes/remove_nops.hpp"
#include "passes/replace_incommutable_patterns.hpp"
#include "passes/set_tile_dim.hpp"
#include "passes/squeeze_to_reshape.hpp"
#include "passes/dataformat.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace tt {

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;
using NodeId = graphlib::NodeId;
using PortId = graphlib::PortId;

void lower_reshape(Graph *, graphlib::OpNode *node)
{
    graphlib::OpType op_type = node->op_type();
    TT_ASSERT(op_type.attr.size() == 4);
    op_type.forge_attrs = {
        {"w", std::get<int>(op_type.attr[0])},
        {"z", std::get<int>(op_type.attr[1])},
        {"r", std::get<int>(op_type.attr[2])},
        {"c", std::get<int>(op_type.attr[3])},
    };
    node->change_op_type(op_type);
}

// *****************************************************************
//  ************************** Main APIs **************************
// *****************************************************************

// ********** Run post initial graph passes **********
std::tuple<std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>>, passes::FractureChipIdAssignments>
run_post_initial_graph_passes(graphlib::Graph *graph, py::object compiler_cfg_object, passes::FractureGroups const &fracture_groups)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "INITIAL");
    passes::generate_initial_flops_estimate(graph);
    passes::decompose_nd_reshape_split(graph);
    passes::limit_to_4d_reshape(graph);
    passes::erase_unnecessary_4d_tm_sequence(graph);
    passes::fuse_pad_conv2d(graph);
    passes::explicate_unsqueeze(graph);
    passes::fuse_conv2d_bias(graph);

    auto inserted_node_id_mapping = decompose_tt_forge_graph(graph, "get_f_forge_decompose", compiler_cfg);
    auto chip_id_assignments = passes::fracture(graph, fracture_groups);
    return std::make_tuple(inserted_node_id_mapping, chip_id_assignments);
}

void run_optimization_graph_passes(graphlib::Graph *graph)
{
    passes::print_graph(graph, "PRE OPTIMIZE");
    passes::lower_concat_to_runtime_transform(graph);

    passes::bypass_nop_tms(graph);

    // Erase all inverse ops possible. 
    // Then, if no inverse ops are erased, then attempt to insert inverse ops on the output. 
    // Then, if no inverse ops can be inserted on the output, then attempt to insert inverse ops on the input.
    // Then, if no inverse ops can be inserted on the input, then attempt to bypass nop reshapes.
    //        NOTE: The reason we try this last is because nop reshapes may end up being inverses of other ops
    //              and we want to erase them that way first if possible
    // Commuting to input may have introduced clones, so attempt to erase inverse ops again
    // ...

    bool attempt_update = true;
    while (attempt_update)
    {
        passes::hoist_unsqueeze_squeeze_to_reshape(graph);

        bool skip_erase_redundant = false;
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update) {
            attempt_update = passes::insert_inverse_on_outputs(graph);
            if (attempt_update)
                skip_erase_redundant = true;
        }
        if (not attempt_update)
            attempt_update = passes::insert_inverse_on_inputs(graph);
        if (not attempt_update) {
            attempt_update = passes::insert_inverse_on_downstream_tms(graph);
            if (attempt_update)
                skip_erase_redundant = true;
        }
        if (not attempt_update)
            attempt_update = passes::replace_incommutable_patterns(graph);

        // These passes erase tms for non-inverse reasons. Usually we are fine with this.
        // However, we might insert tms on top or under of other tms for the purpose of erasing other inverse ops.
        // Skip in that case
        if (not skip_erase_redundant) {
            if (not attempt_update)
                attempt_update = passes::erase_consecutive_reshape(graph, true);

            // TODO: Figure out if this is needed. (Issue #152)
            // if (not attempt_update)
            //     attempt_update = passes::fuse_tm_sequences(graph);

            passes::bypass_nop_tms(graph);
        }
    }
    passes::move_tm_through_requantize(graph);
    recalculate_shapes(graph);

    passes::hoist_transforms_to_inputs(graph);
    passes::erase_consecutive_reshape(graph, true);
    passes::lower_reinterpret_shape(graph);

    passes::fuse_per_channel_ops(graph);
    if (not env_as<bool>("FORGE_DISABLE_CONSTANT_FOLDING"))
        passes::constant_folding(graph);

    passes::move_select_after_matmul_optional(graph);

    // Issue #152
    // passes::fuse_tm_sequences(graph);

    reportify::dump_graph(graph->name(), "post_erase_inverse_ops", graph);
}

std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_optimize_decompose_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "POST_OPTIMIZE");
    auto inserted_node_id_mapping = decompose_tt_forge_graph(graph, "get_f_forge_decompose_post_optimize", compiler_cfg);

    return inserted_node_id_mapping;
}

// ********** Run post autograd graph passes **********
std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_autograd_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "POST_AUTOGRAD");
    lower_bwd_gather_ops(graph);
    return decompose_tt_forge_graph(graph, "get_f_forge_decompose_post_autograd", compiler_cfg);
}

// ********** Run pre-lowering passes **********
graphlib::Graph* run_pre_lowering_passes(
    graphlib::Graph *graph,
    const std::optional<DataFormat> default_df_override)
{
    passes::print_graph(graph, "PRE_MLIR");
    // Recalculate shapes, and figure out implicit broadcasts that are missing
    recalculate_shapes(graph);

    // Fuse bias into matmuls
    fuse_bias(graph);

    // Fuse requantize into matmuls
    fuse_requantize(graph);
    
    // Fuse gelu into matmuls
    if (env_as<bool>("FORGE_FUSE_MATMUL_GELU")) {
        fuse_gelu(graph);
    }

    // Manually convert broadcast ops to tms, so insert tile broadcast ops can work generically
    // Note this is not lowering, these are still forge tms
    convert_broadcast_ops_to_tms(graph);

    // Bypass embedding input nops
    bypass_embedding_input_nops(graph);

    //
    // Data formats
    //
    // Apply user overrides
    passes::configure_output_data_formats(graph, default_df_override);

    passes::remove_nops(graph);

    // Recalculate shapes before lowering to MLIR
    recalculate_shapes(graph);

    return graph;
}

// ********** Run lowering passes **********
std::unique_ptr<graphlib::Graph> run_pre_placer_forge_passes(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    std::vector<std::uint32_t> chip_ids,
    const std::vector<std::string> &op_names_dont_fuse,
    const std::vector<std::string> &op_names_manual_fuse,
    const passes::FractureChipIdAssignments &fracture_chip_id_assignments,
    const std::optional<DataFormat> default_df_override,
    const std::optional<DataFormat> default_accumulate_df,
    const bool enable_broadcast_splitting,
    const DataFormat fp32_fallback,
    const MathFidelity default_math_fidelity,
    const bool enable_auto_fusing,
    const int amp_level,
    const bool enable_recompute,
    const bool output_queues_on_host,
    const bool input_queues_on_host,
    const std::vector<std::tuple<std::string, std::string, int>> &insert_queues,
    std::vector<AMPNodeProperties> amp_properties,
    const std::vector<std::string> &op_intermediates_to_save,
    const bool use_interactive_placer,
    bool enable_device_tilize)
{
    log_debug(LogGraphCompiler, "Lowering target device\n{}", device_config);

    passes::print_graph(graph, "PRE_PLACER");

    // Create forge ops / tms
    std::unique_ptr<graphlib::Graph> lowered_graph = lower_to_forge_ops(graph);

    // lower user-defined buffering queues to actual queue types
    lower_to_buffering_queues(lowered_graph.get());

    split_unsupported_gradient_ops(lowered_graph.get(), device_config);
    recalculate_shapes(lowered_graph.get());

    // Remove nops
    remove_nops(lowered_graph.get());

    // Add buffer NOP between host input and ops if there are multiple ops reading from same host input.
    //
    if (input_queues_on_host and env_as<bool>("FORGE_ENABLE_HOST_INPUT_NOP_BUFFERING"))
    {
        fix_host_inputs(lowered_graph.get());
    }

    // Sanitize past cache IOs
    sanitize_past_cache_ios(lowered_graph.get());

    // Remove transposes from srcB
    bool device_supports_tm_on_srcb = false;  // TODO: device descriptor
    if (!device_supports_tm_on_srcb)
        fix_transposes(lowered_graph.get(), device_config);

    // Remove TMs from output node
    fix_tms_on_output(lowered_graph.get());

    // Need to run before fixing ops that require untilize nop back to host
    insert_queues_for_op_intermediates(lowered_graph.get(), op_intermediates_to_save);

    // Add NOPs on ops feeding output that can't do it directly
    if (output_queues_on_host)
    {
        fix_untilized_outputs(lowered_graph.get(), device_config);
    }

    // Replace "buffer" placeholders with NOPs
    replace_buffers_with_nops(lowered_graph.get());

    insert_nop_on_matmul_input(lowered_graph.get());

    if (enable_device_tilize)
    {
        // If true, insert tilize op after activation (input)
        insert_tilize_op_on_input(lowered_graph.get());
    }

    // Recalculate shapes
    recalculate_shapes(lowered_graph.get());

    // Split big broadcasts into multiple smaller ones by adding nops between them
    if (enable_broadcast_splitting)
    {
        split_broadcasts(lowered_graph.get());
    }

    if (env_as<bool>("FORGE_ENABLE_CONSTANT_PRE_BROADCAST"))
    {
        constant_pre_broadcast(lowered_graph.get());
    }

    if (enable_recompute)
    {
        insert_recompute_ops(lowered_graph.get());
    }

    insert_partial_datacopy_tms(lowered_graph.get());

    insert_user_defined_queues(lowered_graph.get(), insert_queues);

    //
    // Data formats
    //
    run_dataformat_passes(
        lowered_graph.get(),
        device_config,
        default_df_override,
        default_accumulate_df,
        fp32_fallback,
        default_math_fidelity,
        amp_level,
        amp_properties);

    return lowered_graph;
}
}  // namespace tt
