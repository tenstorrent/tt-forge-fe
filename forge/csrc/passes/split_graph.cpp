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

static void copy_recompute_operand(
    graphlib::Node* operand, 
    const Graph* fwd_graph, 
    Graph* bwd_graph, 
    const std::unordered_set<std::string>& recompute_ops)
{
    if (!operand || bwd_graph->has_node_with_name(operand->name())) return;

    // Handle input nodes - just copy them
    if (operand->node_type() == graphlib::NodeType::kInput) {
        clone_and_add(operand, bwd_graph);
        return;
    }

    // Copy dependencies first
    for (auto dep : fwd_graph->operands(operand)) {
        if (!dep || !dep->is_forward()) continue;
        
        // Copy if it's marked for recompute or has no intermediate
        bool need_copy = recompute_ops.count(dep->name()) > 0;
        if (!need_copy) {
            bool has_intermediate = false;
            for (auto user : fwd_graph->data_users(dep)) {
                if (user->node_type() == graphlib::NodeType::kOutput) {
                    has_intermediate = true;
                    break;
                }
            }
            need_copy = !has_intermediate;
        }
        
        if (need_copy) {
            copy_recompute_operand(dep, fwd_graph, bwd_graph, recompute_ops);
        }
    }

    // Copy this operand
    clone_and_add(operand, bwd_graph);
    
    // Connect to dependencies
    auto bwd_operand = bwd_graph->get_node_by_name(operand->name());
    for (auto dep : fwd_graph->operands(operand)) {
        if (!dep || !dep->is_forward()) continue;
        
        if (bwd_graph->has_node_with_name(dep->name())) {
            auto bwd_dep = bwd_graph->get_node_by_name(dep->name());
            bwd_graph->add_edge(bwd_dep, bwd_operand);
        }
    }
}

static void use_intermediate_or_create(
    graphlib::Node* operand, 
    graphlib::Node* bwd_node, 
    const Graph* fwd_graph, 
    Graph* bwd_graph)
{
    auto fwd_operand = fwd_graph->get_node_by_name(operand->name());
    if (!fwd_operand) return;

    for (auto user : fwd_graph->data_users(fwd_operand)) {
        if (user->node_type() != graphlib::NodeType::kOutput) continue;

        if (bwd_graph->has_node_with_name(user->name())) {
            auto intermediate = bwd_graph->get_node_by_name(user->name());
            bwd_graph->add_edge(intermediate, bwd_node);
            return;
        }

        // Create intermediate input
        auto intermediate_input = graphlib::create_node<graphlib::InputNode>(
            user->name(), graphlib::InputNodeType::Activation, false);
        intermediate_input->set_shape(user->shape());
        intermediate_input->set_output_df(user->output_df());
        bwd_graph->add_node(std::move(intermediate_input), 0);
        
        auto bwd_intermediate = bwd_graph->get_node_by_name(user->name());
        if (bwd_intermediate) {
            bwd_graph->add_edge(bwd_intermediate, bwd_node);
            return;
        }
    }
}

std::unique_ptr<Graph> extract_backward_graph(
    const Graph *graph, 
    const Graph *fwd_graph, 
    const std::vector<graphlib::Node *> &topo,
    const std::unordered_set<std::string>& recompute_ops)
{
    auto bwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "backward");
    bwd_graph->set_training(graph->training());

    // Add intermediate inputs from forward graph
    for (auto intermediate : fwd_graph->ordered_intermediates()) {
        auto input = graphlib::create_node<graphlib::InputNode>(
            intermediate->name(), graphlib::InputNodeType::Activation, false);
        input->set_shape(intermediate->shape());
        input->set_output_df(intermediate->output_df());
        bwd_graph->add_node(std::move(input), 0);
    }

    // Add forward inputs that have backward users
    for (auto input : graph->nodes_by_type(graphlib::NodeType::kInput)) {
        if (input->is_forward() && has_users_in_bwd(graph, input) && 
            !bwd_graph->has_node_with_name(input->name())) {
            clone_and_add(input, bwd_graph.get());
        }
    }

    // Process backward nodes - simple single pass
    for (auto node : topo) {
        if (!node->is_backward()) continue;

        if (node->node_type() == graphlib::NodeType::kQueue) {
            auto queue_node = node->as<graphlib::QueueNode>();
            TT_ASSERT(queue_node->is_grad_accumulator(), "Expected only grad accumulator queue nodes");

            auto operand = graph->data_operands(queue_node)[0];
            auto output = graphlib::create_node<graphlib::OutputNode>(queue_node->name() + "_grad_accumulator");
            output->set_output_type(graphlib::OutputType::Internal);
            output->set_shape(queue_node->shape());
            output->set_output_df(queue_node->output_df());
            
            auto grad_out = bwd_graph->add_node(std::move(output), 0);
            auto cloned_operand = bwd_graph->get_node_by_name(operand->name());
            TT_ASSERT(cloned_operand, "Expected operand {} to be present in backward graph", operand->name());
            
            bwd_graph->add_edge(cloned_operand, grad_out, 0, 0, graphlib::EdgeType::kData);
            continue;
        }

        clone_and_add(node, bwd_graph.get());
        auto bwd_node = bwd_graph->get_node_by_name(node->name());
        if (!bwd_node) continue;

        // Handle each operand - simple decision tree
        for (auto operand : graph->data_operands(node)) {
            if (bwd_graph->has_node_with_name(operand->name())) {
                // Easy case - operand already exists
                auto bwd_operand = bwd_graph->get_node_by_name(operand->name());
                bwd_graph->add_edge(bwd_operand, bwd_node);
            } else if (operand->is_forward()) {
                // Forward operand - check if it needs recompute
                if (recompute_ops.count(operand->name()) > 0) {
                    auto fwd_operand = fwd_graph->get_node_by_name(operand->name());
                    copy_recompute_operand(fwd_operand, fwd_graph, bwd_graph.get(), recompute_ops);
                    auto bwd_operand = bwd_graph->get_node_by_name(operand->name());
                    bwd_graph->add_edge(bwd_operand, bwd_node);
                } else {
                    // Use intermediate
                    use_intermediate_or_create(operand, bwd_node, fwd_graph, bwd_graph.get());
                }
            }
        }
    }

    // Register module inputs and outputs
    std::vector<graphlib::NodeId> bwd_inputs, bwd_outputs;

    // Add backward inputs
    for (auto input : graph->ordered_module_inputs()) {
        if (input->is_backward() && bwd_graph->has_node_with_name(input->name())) {
            bwd_inputs.push_back(bwd_graph->get_node_by_name(input->name())->id());
        }
    }

    // Add intermediate inputs
    for (auto output : fwd_graph->ordered_module_outputs()) {
        if (bwd_graph->has_node_with_name(output->name())) {
            bwd_inputs.push_back(bwd_graph->get_node_by_name(output->name())->id());
        }
    }

    // Add forward inputs (avoid duplicates, skip parameters)
    for (auto input : graph->ordered_module_inputs()) {
        if (!input->is_forward() || !bwd_graph->has_node_with_name(input->name())) continue;
        
        auto bwd_input = bwd_graph->get_node_by_name(input->name());
        if (bwd_input->node_type() == graphlib::NodeType::kInput && 
            bwd_input->as<graphlib::InputNode>()->is_parameter()) {
            continue;
        }
        
        auto input_id = bwd_input->id();
        if (std::find(bwd_inputs.begin(), bwd_inputs.end(), input_id) == bwd_inputs.end()) {
            bwd_inputs.push_back(input_id);
        }
    }

    // Add backward outputs
    for (auto output : graph->ordered_module_outputs()) {
        if (output->is_backward() && bwd_graph->has_node_with_name(output->name())) {
            bwd_outputs.push_back(bwd_graph->get_node_by_name(output->name())->id());
        }
    }

    bwd_graph->register_module_inputs(bwd_inputs);
    bwd_graph->register_module_outputs(bwd_outputs);

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
