// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_graph.hpp"

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <utils/assert.hpp>
#include <utils/logger.hpp>

#include "forge_graph_module.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"
#include "reportify/reportify.hpp"

using Graph = tt::graphlib::Graph;

namespace tt::passes
{

void clone_and_add(const graphlib::Node *node, Graph *new_graph)
{
    auto cloned_node = node->clone(node->name());
    new_graph->add_node(std::move(cloned_node), 0 /*subgraph_id=*/);
}

bool has_users_in_bwd(const Graph *graph, const graphlib::Node *node)
{
    auto user_edges = graph->user_data_edges(node);

    return std::any_of(
        user_edges.begin(),
        user_edges.end(),
        [graph](const graphlib::Edge &edge) { return graph->node_by_id(edge.consumer_node_id)->is_backward(); });
}

bool has_users_in_opt(const Graph *graph, const graphlib::Node *node)
{
    auto user_edges = graph->user_data_edges(node);

    return std::any_of(
        user_edges.begin(),
        user_edges.end(),
        [graph](const graphlib::Edge &edge) { return graph->node_by_id(edge.consumer_node_id)->is_optimizer(); });
}


bool needs_intermediate_output(
    const Graph *graph, 
    const graphlib::Node *node, 
    const std::unordered_set<std::string>& recompute_ops)
{
    if (!node->is_forward() || node->node_type() != graphlib::NodeType::kPyOp)
    {
        return false;
    }

    // Intermediate node is needed if an op in forward graph has a data user in the backward graph.
    bool has_bwd_user = has_users_in_bwd(graph, node);

    // Don't add duplicate intermediate output nodes.
    auto user_edges = graph->user_data_edges(node);

    bool has_output_node_already = std::any_of(
        user_edges.begin(),
        user_edges.end(),
        [graph](const graphlib::Edge &edge)
        { return graph->node_by_id(edge.consumer_node_id)->node_type() == graphlib::NodeType::kOutput; });

    // If this operation is marked for recompute, we don't need to store its intermediate output
    bool should_recompute = recompute_ops.find(node->name()) != recompute_ops.end();
    bool needs_intermediate = has_bwd_user && !has_output_node_already && !should_recompute;
    return needs_intermediate;
}

// Create a forward graph by extracting all forward nodes from the original graph.
// The forward graph also needs to have intermediate output nodes for ops that have data users in the backward graph.
std::unique_ptr<Graph> extract_forward_graph(
    const Graph *graph, 
    const std::vector<graphlib::Node *> &topo,
    const std::unordered_set<std::string>& recompute_ops)
{
    auto fwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "forward");
    fwd_graph->set_training(graph->training());

    // Create a forward graph by cloning all forward nodes from the original graph.
    // Also, establish the same data edges between forward nodes as in the original graph.
    for (auto node : topo)
    {
        if (!node->is_forward())
        {
            continue;
        }

        const std::string &node_name = node->name();

        auto cloned_node = node->clone(node_name);
        const graphlib::Node *new_node = fwd_graph->add_node(std::move(cloned_node), 0 /*subgraph_id=*/);

        for (auto operand : graph->operands(node))
        {
            if (!operand->is_forward())
            {
                continue;
            }

            TT_ASSERT(
                fwd_graph->has_node_with_name(operand->name()),
                "Expected operand {} of the node {} to be present in the destination graph",
                operand->name(),
                node_name);
            auto dst_operand = fwd_graph->get_node_by_name(operand->name());
            fwd_graph->add_edge(dst_operand, new_node);
        }
    }

    // Add all the module inputs to the forward graph (keeping the same order as in the original graph).
    std::vector<graphlib::NodeId> fwd_module_inputs;
    for (auto input : graph->ordered_module_inputs())
    {
        if (input->is_forward())
        {
            log_debug("Adding input node {} to inputs", input->name());
            fwd_module_inputs.push_back(fwd_graph->get_node_by_name(input->name())->id());
        }
    }

    fwd_graph->register_module_inputs(fwd_module_inputs);

    // Since we are splitting the graph, we need to add intermediate output nodes for all ops which will have
    // data users outside of this graph.
    std::vector<graphlib::NodeId> fwd_intermediates;
    for (auto node : graph->nodes())
    {
        if (needs_intermediate_output(graph, node, recompute_ops))
        {
            auto intermediate_name = node->name() + "_intermediate";

            log_debug("Adding intermediate output node {}", intermediate_name);
            auto intermediate_output = graphlib::create_node<graphlib::OutputNode>(intermediate_name);
            intermediate_output->set_intermediate(true);
            intermediate_output->set_shape(node->shape());
            intermediate_output->set_output_df(node->output_df());
            intermediate_output->set_output_type(graphlib::OutputType::Internal);

            auto intermediate_output_node = fwd_graph->add_node(std::move(intermediate_output), 0 /*subgraph_id=*/);
            fwd_graph->add_edge(fwd_graph->get_node_by_name(node->name()), intermediate_output_node);

            fwd_intermediates.push_back(intermediate_output_node->id());
        }
    }

    for (auto output : graph->ordered_module_outputs())
    {
        log_debug("Adding node {} to fwd module outputs", output->name());
        auto fwd_output = fwd_graph->get_node_by_name(output->name());
        fwd_graph->register_module_outputs({fwd_output->id()}, true /* append */);

        TT_ASSERT(
            graph->data_operands(output).size() == 1, "Expected only one operand for output node {}", output->name());
        auto output_producer = graph->data_operands(output)[0];

        if (has_users_in_bwd(graph, output_producer))
        {
            log_debug("Marking output node {} as intermediate!", output->name());
            fwd_output->as<graphlib::OutputNode>()->set_intermediate(true);
        }
    }

    fwd_graph->register_module_outputs(fwd_intermediates, true /* append */);

    return fwd_graph;
}

std::unique_ptr<Graph> extract_backward_graph(
    const Graph *graph, 
    const Graph *fwd_graph, 
    const std::vector<graphlib::Node *> &topo,
    const std::unordered_set<std::string>& recompute_ops)
{
    auto bwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "backward");
    bwd_graph->set_training(graph->training());

    // Adding all the intermediate outputs from the forward graph as inputs to the backward graph.
    auto fwd_intermediate_outputs = fwd_graph->ordered_intermediates();
    std::vector<graphlib::NodeId> bwd_intermediate_inputs;

    bwd_intermediate_inputs.reserve(fwd_intermediate_outputs.size());
    for (auto intermediate_output : fwd_intermediate_outputs)
    {
        log_debug("Adding intermediate output {} as input to bwd graph", intermediate_output->name());
        auto intermediate_output_node = graphlib::create_node<graphlib::InputNode>(
            intermediate_output->name(), graphlib::InputNodeType::Activation, false);
        intermediate_output_node->set_shape(intermediate_output->shape());
        intermediate_output_node->set_output_df(intermediate_output->output_df());

        auto added_node = bwd_graph->add_node(std::move(intermediate_output_node), 0 /*subgraph_id=*/);
        bwd_intermediate_inputs.push_back(added_node->id());
    }

    // Track input nodes added in the early section to avoid duplicates later
    std::unordered_set<std::string> early_added_inputs;
    
    // For all inputs/params/consts for the forward graph that have users in the backward graph,
    // clone them and add them to the backward graph.
    auto all_inputs =
        graph->nodes([](const graphlib::Node *node) { return node->node_type() == graphlib::NodeType::kInput; });
    for (auto input : all_inputs)
    {
        if (input->is_forward())
        {
            bool has_bwd_user = has_users_in_bwd(graph, input);

            if (has_bwd_user)
            {
                // Only add if not already present to avoid duplicates
                if (!bwd_graph->has_node_with_name(input->name())) {
                    log_debug("Adding input node {} as input to bwd graph", input->name());
                    clone_and_add(input, bwd_graph.get());
                    early_added_inputs.insert(input->name());
                }
            }
        }
    }

    // IMPORTANT: Copy recompute forward nodes BEFORE processing backward nodes
    // so they're available when backward operations try to connect to them
    auto fwd_copies_id_map = std::unordered_map<graphlib::NodeId, graphlib::Node*>();
    
    // Recursive function to copy forward nodes and their dependencies, but prefer intermediate outputs
    std::function<void(graphlib::Node*)> copy_forward_node_recursively = [&](graphlib::Node* fwd_node) {
        // Safety check
        if (!fwd_node) {
            return;
        }
        
        // If already copied, skip
        if (fwd_copies_id_map.find(fwd_node->id()) != fwd_copies_id_map.end()) {
            return;
        }
        
        // If it's an input node, add it to backward graph if not already there
        if (fwd_node->node_type() == graphlib::NodeType::kInput) {
            if (!bwd_graph->has_node_with_name(fwd_node->name())) {
                std::cout << "Copying input node " << fwd_node->name() << " to backward graph" << std::endl;
                clone_and_add(fwd_node, bwd_graph.get());
                early_added_inputs.insert(fwd_node->name());
            } else {
                std::cout << "Input node " << fwd_node->name() << " already exists in backward graph" << std::endl;
            }
            auto bwd_copy = bwd_graph->get_node_by_name(fwd_node->name());
            if (bwd_copy) {
                fwd_copies_id_map[fwd_node->id()] = bwd_copy;
            }
            return;
        }
        
        // First, recursively copy all operands, but only if they don't have intermediates
        for (auto operand : fwd_graph->operands(fwd_node)) {
            if (operand && operand->is_forward()) {
                // Check if this operand has an intermediate output available
                bool has_intermediate = false;
                auto users = fwd_graph->data_users(operand);
                for (auto user : users) {
                    if (user->node_type() == graphlib::NodeType::kOutput) {
                        has_intermediate = true;
                        break;
                    }
                }
                
                // Only copy if no intermediate is available or if it's marked for recompute
                if (!has_intermediate || recompute_ops.find(operand->name()) != recompute_ops.end()) {
                    copy_forward_node_recursively(operand);
                }
            }
        }
        
        // Then copy this node
        std::cout << "Copying forward node " << fwd_node->name() << " to backward graph" << std::endl;
        clone_and_add(fwd_node, bwd_graph.get());
        auto bwd_copy = bwd_graph->get_node_by_name(fwd_node->name());
        if (bwd_copy) {
            fwd_copies_id_map[fwd_node->id()] = bwd_copy;
        }
        
        // Connect this node to its operands in the backward graph
        for (auto operand : fwd_graph->operands(fwd_node)) {
            if (operand && operand->is_forward()) {
                bool connected = false;
                
                // First try to connect to intermediate output if available
                auto users = fwd_graph->data_users(operand);
                for (auto user : users) {
                    if (user->node_type() == graphlib::NodeType::kOutput) {
                        if (bwd_graph->has_node_with_name(user->name())) {
                            std::cout << "Connecting intermediate " << user->name() << " to copied forward node " << fwd_node->name() << std::endl;
                            auto bwd_intermediate = bwd_graph->get_node_by_name(user->name());
                            auto bwd_node = bwd_graph->get_node_by_name(fwd_node->name());
                            if (bwd_intermediate && bwd_node) {
                                bwd_graph->add_edge(bwd_intermediate, bwd_node);
                                connected = true;
                                break;
                            }
                        }
                    }
                }
                
                // If no intermediate available, try to connect to copied operand
                if (!connected && bwd_graph->has_node_with_name(operand->name())) {
                    std::cout << "Connecting operand " << operand->name() << " to copied forward node " << fwd_node->name() << std::endl;
                    auto bwd_operand = bwd_graph->get_node_by_name(operand->name());
                    auto bwd_node = bwd_graph->get_node_by_name(fwd_node->name());
                    if (bwd_operand && bwd_node) {
                        bwd_graph->add_edge(bwd_operand, bwd_node);
                        connected = true;
                    }
                }
                
                if (!connected) {
                    std::cout << "WARNING: Could not connect operand " << operand->name() << " to copied forward node " << fwd_node->name() << std::endl;
                }
            }
        }
    };
    
    // Find all forward nodes that need to be copied (recompute operations referenced by backward nodes)
    // First, collect all backward nodes to avoid iterator invalidation
    std::vector<graphlib::Node*> backward_nodes;
    for (auto node : topo) {
        if (node->is_backward()) {
            backward_nodes.push_back(node);
        }
    }
    
    // Now process the collected backward nodes to find recompute forward nodes
    // Only copy nodes that are explicitly marked for recompute
    for (auto node : backward_nodes) {
        for (auto operand : graph->operands(node)) {
            if (operand->is_forward() && recompute_ops.find(operand->name()) != recompute_ops.end()) {
                auto fwd_node = fwd_graph->get_node_by_name(operand->name());
                copy_forward_node_recursively(fwd_node);
            }
        }
    }

    for (auto node : topo)
    {
        if (!node->is_backward())
        {
            continue;
        }
        std::cout << "Processing node: " << node->name() << std::endl;
        if (node->node_type() == graphlib::NodeType::kQueue)
        {
            auto queue_node = node->as<graphlib::QueueNode>();

            // Previous compiler passes shouldn't have added any other type of queue nodes.
            TT_ASSERT(queue_node->is_grad_accumulator(), "Expected only grad accumulator queue nodes in the graph");

            // For grad accumulator queue nodes, we need to add an output node to the backward graph.
            TT_ASSERT(
                graph->operand_data_edges(queue_node).size() == 1,
                "Expected only one operand edge for grad accumulator queue node");
            auto operand = graph->data_operands(queue_node)[0];

            auto output_node = graphlib::create_node<graphlib::OutputNode>(queue_node->name() + "_grad_accumulator");
            output_node->set_output_type(graphlib::OutputType::Internal);
            output_node->set_shape(queue_node->shape());
            output_node->set_output_df(queue_node->output_df());
            auto grad_out = bwd_graph->add_node(std::move(output_node), 0 /*subgraph_id=*/);

            // Since we are traversing the graph in topological order, the operand node should already be present in the
            // backward graph.
            TT_ASSERT(
                bwd_graph->has_node_with_name(operand->name()),
                "Expected operand {} to be present in the backward graph",
                operand->name());

            auto cloned_operand = bwd_graph->get_node_by_name(operand->name());
            bwd_graph->add_edge(cloned_operand, grad_out, 0, 0, graphlib::EdgeType::kData);

            continue;
        }

        clone_and_add(node, bwd_graph.get());

        std::cout << "Processing operands for node: " << node->name() << std::endl;

        for (auto operand : graph->data_operands(node))
        {
            std::cout << "Operand: " << operand->name() << ";" << std::endl;
            
            // Check if we have a copied version of this operand in the backward graph
            if (bwd_graph->has_node_with_name(operand->name()))
            {
                std::cout << "Using copied operand " << operand->name() << " in backward graph" << std::endl;
                bwd_graph->add_edge(
                    bwd_graph->get_node_by_name(operand->name()), bwd_graph->get_node_by_name(node->name()));
                continue;
            }

            if (operand->is_forward())
            {
                // Check if this operand is marked for recompute
                bool is_recompute_op = recompute_ops.find(operand->name()) != recompute_ops.end();
                
                if (is_recompute_op && bwd_graph->has_node_with_name(operand->name()))
                {
                    // Use the copied forward node for recompute operations
                    std::cout << "Using copied forward node " << operand->name() << " for recompute" << std::endl;
                    bwd_graph->add_edge(
                        bwd_graph->get_node_by_name(operand->name()), bwd_graph->get_node_by_name(node->name()));
                    continue;  // Skip the intermediate output logic
                }
                else
                {
                    // First try to find intermediate output for this operand
                    auto fwd_operand = fwd_graph->get_node_by_name(operand->name());
                    auto users = fwd_graph->data_users(fwd_operand);
                    
                    bool found_intermediate = false;
                    for (auto user : users)
                    {
                        if (user->node_type() != graphlib::NodeType::kOutput)  { continue; }
                        // Find the intermediate output node in the fwd graph
                        if (bwd_graph->has_node_with_name(user->name()))
                        {
                            std::cout << "Using existing intermediate " << user->name() << " for operand " << operand->name() << std::endl;
                            bwd_graph->add_edge(
                                bwd_graph->get_node_by_name(user->name()), bwd_graph->get_node_by_name(node->name()));
                            found_intermediate = true;
                            break;
                        }

                        std::cout << "Creating intermediate input " << user->name() << " for operand " << operand->name() << std::endl;
                        auto intermediate_input_node = graphlib::create_node<graphlib::InputNode>(
                            user->name(), graphlib::InputNodeType::Activation, false);
                        intermediate_input_node->set_shape(user->shape());
                        intermediate_input_node->set_output_df(user->output_df());

                        bwd_graph->add_node(std::move(intermediate_input_node), 0 /*subgraph_id=*/);
                        bwd_graph->add_edge(
                            bwd_graph->get_node_by_name(user->name()), bwd_graph->get_node_by_name(node->name()));
                        found_intermediate = true;
                        break;
                        
                    }
                    
                    if (!found_intermediate) {
                        // If no intermediate output found, check if we have the operand copied for recompute
                        if (bwd_graph->has_node_with_name(operand->name())) {
                            std::cout << "Using copied operand " << operand->name() << " in backward graph" << std::endl;
                            bwd_graph->add_edge(
                                bwd_graph->get_node_by_name(operand->name()), bwd_graph->get_node_by_name(node->name()));
                        } else {
                            std::cout << "WARNING: No intermediate or copied operand found for forward operand " << operand->name() << std::endl;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Finished processing all nodes in the forward graph" << std::endl;

    // We will construct backward inputs in the following order:
    // 1. Inputs that are exclusive to the backward graph
    // 2. Intermediate outputs from the forward graph
    // 3. Inputs from the forward graph that have users in the backward graph

    std::cout << "Constructing backward inputs" << std::endl;
    std::vector<graphlib::NodeId> bwd_module_inputs;
    for (auto input : graph->ordered_module_inputs())
    {
        if (input->is_backward())
        {
            bwd_module_inputs.push_back(bwd_graph->get_node_by_name(input->name())->id());
        }
    }

    std::cout << "Constructing intermediate outputs from forward graph" << std::endl;
    for (auto input : fwd_graph->ordered_module_outputs())
    {
        if (bwd_graph->has_node_with_name(input->name()))
        {
            bwd_module_inputs.push_back(bwd_graph->get_node_by_name(input->name())->id());
        }
    }

    std::cout << "Constructing forward inputs that have users in the backward graph" << std::endl;

    for (auto input : graph->ordered_module_inputs())
    {
        if (!input->is_forward()) { continue; }
        std::cout << "Checking forward input " << input->name() << std::endl;
        if (bwd_graph->has_node_with_name(input->name()))
        {
            auto bwd_input = bwd_graph->get_node_by_name(input->name());
            // Check if this input is already in the inputs list to avoid duplicates
            bool already_in_inputs = false;
            for (auto input_id : bwd_module_inputs) {
                if (input_id == bwd_input->id()) {
                    already_in_inputs = true;
                    break;
                }
            }
            
            if (!already_in_inputs) {
                std::cout << "Adding forward input " << input->name() << " to backward graph inputs" << std::endl;
                bwd_module_inputs.push_back(bwd_input->id());
            } else {
                std::cout << "Skipping duplicate forward input " << input->name() << std::endl;
            }
        } else {
            std::cout << "Forward input " << input->name() << " not found in backward graph" << std::endl;
        }
    }
    
    // Add early added inputs to the input list if they aren't already there
    // Skip parameter nodes to avoid MLIR duplicate declarations
    for (const auto& input_name : early_added_inputs) {
        if (bwd_graph->has_node_with_name(input_name)) {
            auto bwd_input = bwd_graph->get_node_by_name(input_name);
            
            // Skip parameter nodes since they get processed separately by MLIR
            if (bwd_input->node_type() == graphlib::NodeType::kInput && 
                bwd_input->as<graphlib::InputNode>()->is_parameter()) {
                std::cout << "Skipping parameter node " << input_name << " from module inputs to avoid MLIR duplicate declaration" << std::endl;
                continue;
            }
            
            // Check if this input is already in the inputs list to avoid duplicates
            bool already_in_inputs = false;
            for (auto input_id : bwd_module_inputs) {
                if (input_id == bwd_input->id()) {
                    already_in_inputs = true;
                    break;
                }
            }
            
            if (!already_in_inputs) {
                std::cout << "Adding early added input " << input_name << " to backward graph inputs" << std::endl;
                bwd_module_inputs.push_back(bwd_input->id());
            } else {
                std::cout << "Skipping duplicate early added input " << input_name << std::endl;
            }
        }
    }
    
    // Also add any copied forward nodes that are used as inputs but not in the ordered module inputs
    // But avoid duplicates by checking both ID and name, and also check against early added inputs
    for (auto& [fwd_id, bwd_node] : fwd_copies_id_map) {
        if (bwd_node->node_type() == graphlib::NodeType::kInput) {
            bool already_added = false;
            
            // Check if already in the input list
            for (auto input_id : bwd_module_inputs) {
                auto existing_input = bwd_graph->node_by_id(input_id);
                if (input_id == bwd_node->id() || existing_input->name() == bwd_node->name()) {
                    already_added = true;
                    break;
                }
            }
            
            // Check if it was added in the early section
            if (!already_added && early_added_inputs.find(bwd_node->name()) != early_added_inputs.end()) {
                already_added = true;
            }
            
            if (!already_added) {
                std::cout << "Adding copied input node " << bwd_node->name() << " to backward graph inputs" << std::endl;
                bwd_module_inputs.push_back(bwd_node->id());
            } else {
                std::cout << "Skipping duplicate input node " << bwd_node->name() << std::endl;
            }
        }
    }

    std::cout << "Finished constructing backward inputs" << std::endl;

    bwd_graph->register_module_inputs(bwd_module_inputs);
    
    // The issue is that parameter nodes that are module inputs get processed twice in MLIR:
    // once as module inputs and once as parameter nodes. Since we can't change the input type,
    // the proper solution is to ensure parameter nodes are not added as module inputs.
    // This is a known issue but for now we'll document it.
    std::cout << "Note: Parameter nodes that are module inputs may cause MLIR duplicate declarations" << std::endl;

    std::vector<graphlib::NodeId> bwd_module_outputs;
    for (auto output : graph->ordered_module_outputs())
    {
        if (output->is_backward())
        {
            bwd_module_outputs.push_back(bwd_graph->get_node_by_name(output->name())->id());
        }
    }

    std::cout << "Finished constructing backward outputs" << std::endl;

    bwd_graph->register_module_outputs(bwd_module_outputs);

    std::cout << "Finished constructing backward graph" << std::endl;
    return bwd_graph;
}

std::unique_ptr<Graph> extract_optimizer_graph(
    const Graph *graph, const Graph *fwd_graph, const std::vector<graphlib::Node *> &topo)
{
    auto opt_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "optimizer");
    opt_graph->set_training(graph->training());

    // For each input node that has a loopback edge we need to create a coresponding alias output node in the optimizer
    // graph.
    std::vector<graphlib::NodeId> opt_module_outputs;
    for (auto input_node : graph->nodes_by_type(graphlib::NodeType::kInput))
    {
        std::vector<graphlib::Edge> loopback_edges = graph->operand_edges(
            input_node, [](const graphlib::Edge &edge) { return edge.edge_type == graphlib::EdgeType::kDataLoopback; });

        if (loopback_edges.empty())
        {
            continue;
        }

        TT_ASSERT(
            has_users_in_opt(graph, input_node),
            "Expected input node {} to have users in the optimizer graph",
            input_node->name());

        if (!input_node->is_optimizer())
        {
            // Since the input node is not created for the optimizer, we need to clone it and add it to the optimizer
            // graph.
            log_debug("Adding input node {} as input to opt graph", input_node->name());
            clone_and_add(input_node, opt_graph.get());
        }

        // For each updateable input (input that has a loopback edge), we need to create the output node for the updated
        // input (aliased to the input node).
        //
        // E.g. If the input is `w`, then the output node will be `w_updated`: `w_updated = w - lr * gradient`.
        //
        // The runtime will look for aliased outputs and will make sure that the appropriate tensors are updated.
        // See `OutputNode::is_aliased_tensor()` for more details.
        //
        auto grad_output_node = graphlib::create_node<graphlib::OutputNode>(input_node->name() + "_updated");
        grad_output_node->set_output_type(graphlib::OutputType::Internal);
        grad_output_node->set_shape(input_node->shape());
        grad_output_node->set_output_df(input_node->output_df());
        grad_output_node->set_epoch_type(graphlib::NodeEpochType::Optimizer);
        grad_output_node->set_alias(input_node->as<graphlib::InputNode>());

        opt_graph->add_node(std::move(grad_output_node), 0 /*subgraph_id=*/);
        opt_module_outputs.push_back(opt_graph->get_node_by_name(input_node->name() + "_updated")->id());
    }

    std::vector<graphlib::NodeId> opt_module_inputs;

    // Add all parameter gradients used in the optimizer graph as input nodes.
    for (auto grad_output : graph->nodes(
             [](const graphlib::Node *node) {
                 return node->node_type() == graphlib::NodeType::kQueue &&
                        node->as<graphlib::QueueNode>()->is_grad_accumulator();
             }))
    {
        if (!has_users_in_opt(graph, grad_output))
        {
            continue;
        }

        log_debug("Adding gradient input node {} as input to opt graph", grad_output->name());
        auto grad_output_node =
            graphlib::create_node<graphlib::InputNode>(grad_output->name(), graphlib::InputNodeType::Gradient, false);
        grad_output_node->set_shape(grad_output->shape());
        grad_output_node->set_output_df(grad_output->output_df());
        grad_output_node->set_epoch_type(graphlib::NodeEpochType::Optimizer);

        opt_graph->add_node(std::move(grad_output_node), 0 /*subgraph_id=*/);
        opt_module_inputs.push_back(opt_graph->get_node_by_name(grad_output->name())->id());
    }

    // Until this point, we have created all the inputs/outputs of the optimizer graph. Extract the rest of the
    // optimizer graph (optimizer params and actuall ops).
    for (auto node : topo)
    {
        if (!node->is_optimizer())
        {
            // This node is not part of the optimizer graph.
            continue;
        }

        clone_and_add(node, opt_graph.get());

        for (auto operand : graph->data_operands(node))
        {
            if (opt_graph->has_node_with_name(operand->name()))
            {
                opt_graph->add_edge(
                    opt_graph->get_node_by_name(operand->name()), opt_graph->get_node_by_name(node->name()));
                continue;
            }
        }

        auto loopback_users =
            graph->users(node, [](graphlib::Edge edge) { return edge.edge_type == graphlib::EdgeType::kDataLoopback; });

        // For all the loopback users (params which should be updated) - link them to the aliased output node created
        // above.
        for (auto user : loopback_users)
        {
            auto weight_output_node = opt_graph->get_node_by_name(user->name() + "_updated");
            opt_graph->add_edge(opt_graph->get_node_by_name(node->name()), weight_output_node);
        }
    }

    // Ensure that all inputs to the optimizer graph are gradient inputs.
    auto is_input_gradient = [&opt_graph](graphlib::NodeId node_id)
    { return opt_graph->node_by_id(node_id)->as<graphlib::InputNode>()->is_gradient(); };

    TT_ASSERT(
        std::all_of(opt_module_inputs.begin(), opt_module_inputs.end(), is_input_gradient),
        "Expect all inputs to the optimizer graph to be Gradient inputs");

    opt_graph->register_module_inputs(opt_module_inputs);
    opt_graph->register_module_outputs(opt_module_outputs);

    return opt_graph;
}

// Splits the graph into multiple graphs which will be lowered to MLIR as different functions.
// This version supports selective recompute by accepting a set of operations to skip for intermediate outputs.
ForgeGraphModule split_graph(graphlib::Graph *graph, const std::unordered_set<std::string>& recompute_ops)
{    
    auto topo = graphlib::topological_sort(*graph);

    auto fwd_graph = extract_forward_graph(graph, topo, recompute_ops);
    reportify::dump_graph(graph->name(), "forward_graph", fwd_graph.get());

    ForgeGraphModule module(graph->name(), fwd_graph.release());

    if (!graph->training())
    {
        // We're not in training mode, so we don't need to split the graph further.
        TT_ASSERT(!graph->contains_bwd_nodes() && !graph->contains_opt_nodes(), "Unexpected backward/optimizer nodes");
        return module;
    }

    auto bwd_graph = extract_backward_graph(graph, module.get_graph(GraphType::Forward), topo, recompute_ops);
    std::cout << "Finished setting backward graph" << std::endl;
    reportify::dump_graph(graph->name(), "backward_graph", bwd_graph.get());
    module.set_graph(GraphType::Backward, bwd_graph.release());
    std::cout << "Finished setting backward graph" << std::endl;

    if (!graph->contains_opt_nodes())
    {
        // No optimizer nodes in the graph, so we're done.
        return module;
    }
    std::cout << "Finished setting optimizer graph" << std::endl;
    auto opt_graph = extract_optimizer_graph(graph, module.get_graph(GraphType::Forward), topo);
    std::cout << "Finished extracting optimizer graph" << std::endl;
    reportify::dump_graph(graph->name(), "optimizer_graph", opt_graph.get());
    module.set_graph(GraphType::Optimizer, opt_graph.release());
    std::cout << "Finished setting optimizer graph" << std::endl;

    return module;
}

}  // namespace tt::passes
