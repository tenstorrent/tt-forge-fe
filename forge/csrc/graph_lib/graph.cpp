// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph.hpp"

#include <algorithm>
#include <iostream>

#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"

namespace tt
{

namespace graphlib
{

Graph::Graph(IRLevel ir_level) : name_("default_graph"), unique_id_(-1), ir_level_(ir_level) {}
Graph::Graph(IRLevel ir_level, std::string name, int unique_id) :
    name_(name), unique_id_(unique_id), ir_level_(ir_level)
{
}

Node *Graph::get_node_by_name(const std::string &name, bool raise_exception) const
{
    if (!this->has_node_with_name(name))
    {
        if (raise_exception)
        {
            throw std::runtime_error(
                "Graph " + std::to_string(this->id()) + " does not contain node with name: " + name + "\n");
        }
        else
        {
            return nullptr;
        }
    }

    NodeId node_id = this->node_name_to_node_id_.at(name);

    if (!this->has_node_with_id(node_id))
    {
        if (raise_exception)
        {
            throw std::runtime_error(
                "Graph " + std::to_string(this->id()) + " does not contain node with id: " + std::to_string(node_id) +
                "\n");
        }
        else
        {
            return nullptr;
        }
    }

    return this->nodes_map_raw_.at(node_id);
}

NodeId Graph::last_node_id_assigned_ = 0;
GraphId Graph::last_graph_id_assigned_ = 0;
std::unordered_set<GraphId> Graph::assigned_graph_ids_ = std::unordered_set<GraphId>();

Graph *Graph::clone(Graph *cloned_graph) const
{
    const Graph *graph = this;
    if (graph == nullptr)
    {
        return nullptr;
    }

    if (cloned_graph == nullptr)
    {
        GraphId cloned_graph_id = Graph::generate_unique_graph_id();
        std::string cloned_graph_name = graph->name() + "_" + std::to_string(cloned_graph_id);
        cloned_graph = new Graph(ir_level_, cloned_graph_name, cloned_graph_id);
    }

    std::unordered_map<NodeId, NodeId> node_to_cloned_node;
    for (const auto &[node_id, node] : this->nodes_map_raw_)
    {
        auto cloned_node =
            cloned_graph->add_node(node->clone(), graph->get_subgraph_id_for_node(node->id()), node->id());
        node_to_cloned_node[node->id()] = cloned_node->id();
    }
    for (const auto &[node_id, node_operand_edges] : this->operands_map_)
    {
        for (const auto &operand_edge : node_operand_edges)
        {
            NodeId cloned_producer_node_id = node_to_cloned_node.at(operand_edge.producer_node_id);
            NodeId cloned_consumer_node_id = node_to_cloned_node.at(operand_edge.consumer_node_id);
            // Node *cloned_producer = cloned_graph->node_by_id(cloned_producer_node_id);
            // Node *cloned_consumer = cloned_graph->node_by_id(cloned_consumer_node_id);

            Edge new_edge = Edge(
                cloned_producer_node_id,
                operand_edge.producer_output_port_id,
                cloned_consumer_node_id,
                operand_edge.consumer_input_port_id,
                operand_edge.edge_type);

            cloned_graph->add_edge(new_edge);
            cloned_graph->copy_edge_attributes(operand_edge, new_edge, this);
        }
    }

    for (NodeId id : ordered_module_input_node_ids_)
    {
        cloned_graph->ordered_module_input_node_ids_.push_back(node_to_cloned_node.at(id));
    }
    for (NodeId id : ordered_module_output_node_ids_)
    {
        cloned_graph->ordered_module_output_node_ids_.push_back(node_to_cloned_node.at(id));
    }
    for (NodeId id : ordered_module_target_node_ids_)
    {
        cloned_graph->ordered_module_target_node_ids_.push_back(node_to_cloned_node.at(id));
    }

    cloned_graph->set_microbatch(get_microbatch());

    return cloned_graph;
}

// Set of helper functions for supporting GraphTraversalContext,
// utility context which can treat some nodes and edges as visible/invisible
// for custom graph traversal.
//
// In custom context edge is visible if it is not ignored
// and if nodes it is connected to are both visible.
//
bool Graph::is_edge_visible(const Edge &edge) const
{
    // Optimize early out for common case.
    //
    if (nullptr == node_traversal_context_ and nullptr == virtual_node_traversal_context_ and
        virtual_nodes_.size() == 0)
    {
        return true;
    }

    // Edge marked as ignored in GraphTraversalContext.
    //
    if (ignored_edges_traversal_context_ and ignored_edges_traversal_context_->count(edge) > 0)
    {
        return false;
    }

    NodeId producer_node_id = edge.producer_node_id;
    NodeId consumer_node_id = edge.consumer_node_id;
    Node *producer_node = node_by_id(producer_node_id);
    Node *consumer_node = node_by_id(consumer_node_id);

    return (producer_node and consumer_node and is_node_visible(producer_node) and is_node_visible(consumer_node));
}

// Node is visible if it is not virtual or if it is present in GraphTraversalContext.
//
bool Graph::is_node_visible(const Node *node) const
{
    // Optimize early out for common case.
    //
    if (nullptr == node_traversal_context_ and nullptr == virtual_node_traversal_context_ and
        virtual_nodes_.size() == 0)
    {
        return true;
    }
    else
    {
        if (!(is_node_virtual(node)))
        {
            // Regular nodes can be filtered only by node_traversal_context_.
            //
            return node_traversal_context_ == nullptr or node_traversal_context_->count(node) > 0;
        }
        else
        {
            // Virtual nodes can be filtered by both node_traversal_context_ and virtual_node_traversal_context_.
            //
            return (virtual_node_traversal_context_ and virtual_node_traversal_context_->count(node) > 0) or
                   (node_traversal_context_ and node_traversal_context_->count(node) > 0);
        }
    }

    return false;
}

// Tracking virtual nodes.
//
void Graph::mark_node_virtual(const Node *node) { virtual_nodes_.insert(node->id()); }

void Graph::mark_node_persisted(const Node *node)
{
    TT_ASSERT(virtual_nodes_.count(node->id()) > 0);
    virtual_nodes_.erase(node->id());
}

bool Graph::is_node_virtual(const Node *node) const { return virtual_nodes_.count(node->id()) > 0; }

bool Graph::is_graph_traversal_context_set() const
{
    return nullptr != virtual_node_traversal_context_ or ignored_edges_traversal_context_ != nullptr;
}

std::vector<Edge> Graph::edges(EdgeType edge_type) const
{
    std::vector<Edge> queried_edges;
    for (const auto &[node_id, operand_edges] : this->operands_map_)
    {
        for (const Edge &edge : operand_edges)
        {
            if (edge.edge_type == edge_type)
            {
                queried_edges.push_back(edge);
            }
        }
    }
    return queried_edges;
}

std::unordered_set<Edge> &Graph::operand_edges_set(NodeId node_id) { return this->operands_map_.at(node_id); }
const std::unordered_set<Edge> &Graph::operand_edges_set(const Node *node) const
{
    return this->operands_map_.at(node->id());
}

std::vector<Edge> Graph::operand_edges(const Node *node, std::function<bool(Edge)> edge_filter) const
{
    std::vector<Edge> operand_edges;
    auto sort_on_consumer_input_port = [](const Edge &a, const Edge &b) -> bool
    { return a.consumer_input_port_id < b.consumer_input_port_id; };
    for (auto operand_edge : this->operand_edges_set(node))
    {
        if (edge_filter(operand_edge) and is_edge_visible(operand_edge))
        {
            operand_edges.push_back(operand_edge);
        }
    }
    std::sort(std::begin(operand_edges), std::end(operand_edges), sort_on_consumer_input_port);
    return operand_edges;
}

std::vector<Edge> Graph::operand_data_edges(const Node *node, std::function<bool(Edge)> edge_filter) const
{
    return operand_edges(
        node,
        [edge_filter](Edge edge) {
            return edge_filter(edge) and
                   (edge.edge_type == EdgeType::kData or edge.edge_type == EdgeType::kDataLoopback);
        });
}

std::unordered_set<Edge> &Graph::user_edges_set(NodeId node_id) { return this->users_map_.at(node_id); }

std::vector<Edge> Graph::edges(const Node *node, std::function<bool(Edge)> edge_filter) const
{
    std::vector<Edge> operand_edges = this->operand_edges(node, edge_filter);
    std::vector<Edge> user_edges = this->user_edges(node, edge_filter);

    std::vector<Edge> all_edges;
    all_edges.reserve(operand_edges.size() + user_edges.size());
    all_edges.insert(all_edges.end(), operand_edges.begin(), operand_edges.end());
    all_edges.insert(all_edges.end(), user_edges.begin(), user_edges.end());
    return all_edges;
}

std::vector<Edge> Graph::user_edges(const Node *node, std::function<bool(Edge)> edge_filter) const
{
    std::vector<Edge> user_edges;

    // user-edge sorting is <producer_output_port_id, consumer_input_port_id, edge_creation_id>
    // in that exact order.
    auto edge_sort = [](const Edge &a, const Edge &b) -> bool
    {
        if (a.producer_output_port_id != b.producer_output_port_id)
        {
            return a.producer_output_port_id < b.producer_output_port_id;
        }
        if (a.consumer_input_port_id != b.consumer_input_port_id)
        {
            return a.consumer_input_port_id < b.consumer_input_port_id;
        }
        if (a.edge_creation_id != b.edge_creation_id)
        {
            return a.edge_creation_id < b.edge_creation_id;
        }
        return false;
    };

    for (auto user_edge : this->user_edges_set(node))
    {
        if (edge_filter(user_edge) and is_edge_visible(user_edge))
        {
            user_edges.push_back(user_edge);
        }
    }
    std::sort(std::begin(user_edges), std::end(user_edges), edge_sort);
    return user_edges;
}

std::vector<Edge> Graph::user_data_edges(const Node *node, std::function<bool(Edge)> edge_filter) const
{
    return user_edges(
        node,
        [edge_filter](Edge edge) {
            return edge_filter(edge) and
                   (edge.edge_type == EdgeType::kData or edge.edge_type == EdgeType::kDataLoopback);
        });
}

const std::unordered_set<Edge> &Graph::user_edges_set(const Node *node) const
{
    return this->users_map_.at(node->id());
}

std::unordered_set<NodeId> Graph::node_ids()
{
    std::unordered_set<NodeId> node_id_set;
    const auto &node_ids = this->nodes_map_;
    for (auto const &kv_pair : node_ids)
    {
        node_id_set.insert(kv_pair.first);
    }
    return node_id_set;
}

std::vector<Node *> Graph::operands(const Node *node) const
{
    std::vector<Node *> operand_nodes;
    std::vector<Edge> operand_edges = this->operand_edges(node);

    for (auto &operand_edge : operand_edges)
    {
        NodeId producer_node_id = operand_edge.producer_node_id;
        Node *producer_node = node_by_id(producer_node_id);
        operand_nodes.push_back(producer_node);
    }
    return operand_nodes;
}

std::vector<Node *> Graph::data_operands(const Node *node) const
{
    std::vector<Node *> operand_nodes;
    std::vector<Edge> operand_edges = this->operand_data_edges(node);

    for (auto &operand_edge : operand_edges)
    {
        NodeId producer_node_id = operand_edge.producer_node_id;
        Node *producer_node = node_by_id(producer_node_id);
        operand_nodes.push_back(producer_node);
    }
    return operand_nodes;
}

std::vector<Edge> Graph::get_edges(const Node *producer, const Node *consumer) const
{
    std::vector<Edge> edges_btw_producer_consumer;
    std::vector<Edge> user_edges = this->user_data_edges(producer);
    for (const Edge &edge : user_edges)
    {
        if (edge.consumer_node_id == consumer->id())
        {
            edges_btw_producer_consumer.push_back(edge);
        }
    }
    return edges_btw_producer_consumer;
}

std::pair<PortId, int> Graph::output_port_and_index_for_data_user_port(const Node *node, Edge user_edge) const
{
    std::vector<Node *> user_nodes;
    std::vector<Edge> user_edges = this->user_data_edges(node);

    // TODO: Add graph method that takes src/sink pair and returns edge.
    NodeId producer_output_port_id = -1;
    for (auto &user_edge_ : user_edges)
    {
        if (user_edge_ != user_edge)
        {
            continue;
        }
        producer_output_port_id = user_edge.producer_output_port_id;
        break;
    }
    assert(producer_output_port_id != -1);

    std::pair<PortId, int> result = {producer_output_port_id, 0};
    // Important: The order of the user_edges is important - must be deterministic!
    for (auto &user_edge_ : user_edges)
    {
        if (user_edge_.producer_output_port_id != producer_output_port_id)
        {
            continue;
        }
        if (user_edge_ == user_edge)
        {
            break;
        }
        result.second += 1;
    }
    return result;
}

std::vector<std::pair<PortId, int>> Graph::output_port_and_index_for_consumer(
    const Node *producer, const Node *consumer) const
{
    std::vector<std::pair<PortId, int>> output_port_and_index_for_consumer;
    std::vector<Edge> user_edges = this->user_edges(producer);

    std::unordered_map<PortId, int> producer_output_port_id_to_current_index;
    for (const auto &edge : user_edges)
    {
        PortId producer_output_port_id = edge.producer_output_port_id;

        if (edge.consumer_node_id == consumer->id())
        {
            int current_index = producer_output_port_id_to_current_index[producer_output_port_id];
            output_port_and_index_for_consumer.push_back(std::make_pair(producer_output_port_id, current_index));
        }
        producer_output_port_id_to_current_index[producer_output_port_id]++;
    }
    return output_port_and_index_for_consumer;
}

std::vector<Edge> Graph::user_data_edges_for_operand_port(const Node *node, PortId port_id) const
{
    std::vector<Edge> result;
    for (Edge consumer_edge : this->user_data_edges(node))
    {
        if (consumer_edge.producer_output_port_id != port_id)
        {
            continue;
        }
        result.push_back(consumer_edge);
    }
    return result;
}

std::vector<Node *> Graph::users(const Node *node, std::function<bool(Edge)> edge_filter) const
{
    std::vector<Node *> user_nodes;

    for (auto &user_edge : this->user_edges(node, edge_filter))
    {
        NodeId consumer_node_id = user_edge.consumer_node_id;
        Node *consumer_node = node_by_id(consumer_node_id);
        user_nodes.push_back(consumer_node);
    }
    return user_nodes;
}

std::vector<Node *> Graph::data_users(const Node *node) const
{
    std::vector<Node *> user_nodes;

    for (auto &user_edge : this->user_edges(node))
    {
        if (user_edge.edge_type != EdgeType::kData)
        {
            continue;
        }

        NodeId consumer_node_id = user_edge.consumer_node_id;
        Node *consumer_node = node_by_id(consumer_node_id);
        user_nodes.push_back(consumer_node);
    }
    return user_nodes;
}

std::unique_ptr<Node> Graph::remove_node(const NodeId node_id)
{
    for (auto &operand_edge : this->operands_map_[node_id])
    {
        this->users_map_[operand_edge.producer_node_id].erase(operand_edge);
    }
    for (auto &operand_edge : this->users_map_[node_id])
    {
        this->operands_map_[operand_edge.consumer_node_id].erase(operand_edge);
    }

    Node *node_to_delete = this->nodes_map_raw_.at(node_id);
    this->node_name_to_node_id_.erase(node_to_delete->name());

    this->operands_map_.erase(node_id);
    this->users_map_.erase(node_id);
    auto node_unique_ptr = std::move(this->nodes_map_.extract(node_id).mapped());
    this->nodes_map_raw_.erase(node_id);
    this->nodes_.erase(
        std::remove(this->nodes_.begin(), this->nodes_.end(), node_unique_ptr.get()), this->nodes_.end());
    this->virtual_nodes_.erase(node_id);
    node_unique_ptr->set_id(-1);
    return node_unique_ptr;
}

std::unique_ptr<Node> Graph::remove_node(const Node *node) { return remove_node(node->id()); }

const Graph::NodeIdToNodePtr &Graph::nodes_map() const { return this->nodes_map_raw_; }

void Graph::add_edge(const Edge &edge, std::shared_ptr<EdgeAttributes> edge_attributes)
{
    users_map_[edge.producer_node_id].insert(edge);
    operands_map_[edge.consumer_node_id].insert(edge);
    if (edge_attributes)
    {
        edge_to_attr_map_.insert(std::make_pair(edge.unique_id(), edge_attributes));
    }
    else
    {
        edge_to_attr_map_.insert(std::make_pair(edge.unique_id(), EdgeAttributes::create(edge.edge_type)));
    }
}

std::shared_ptr<EdgeAttributes> Graph::get_edge_attributes(const Edge &edge) const
{
    return edge_to_attr_map_.at(edge.unique_id());
}

void Graph::copy_edge_attributes(const Edge &src_edge, const Edge &dest_edge, const Graph *old_graph)
{
    if (old_graph == nullptr)
        old_graph = this;
    auto attr = old_graph->get_edge_attributes(src_edge);
    get_edge_attributes(dest_edge)->copy_from(*attr);
}

// Copy node attributes
void Graph::copy_node_attributes(Node *src, Node *dst)
{
    dst->set_epoch_type(src->get_epoch_type());
    dst->set_output_df(src->output_df());
    if (((dst->node_type() == NodeType::kForgeOp) && (src->node_type() == NodeType::kForgeOp) &&
         src->as<OpNode>()->is_gradient_op()) ||
        ((dst->node_type() == NodeType::kPyOp) && (src->node_type() == NodeType::kPyOp) &&
         src->as<OpNode>()->is_gradient_op()))
        dst->as<OpNode>()->set_gradient_op();
}

void Graph::add_edge(
    const Node &producer,
    const Node &consumer,
    PortId producer_output_port_id,
    PortId consumer_input_port_id,
    EdgeType edge_type)
{
    NodeId producer_node_id = producer.id();
    NodeId consumer_node_id = consumer.id();

    Edge edge(producer_node_id, producer_output_port_id, consumer_node_id, consumer_input_port_id, edge_type);
    this->add_edge(edge);
}

void Graph::add_edge(
    const Node *producer,
    const Node *consumer,
    PortId producer_output_port_id,
    PortId consumer_input_port_id,
    EdgeType edge_type)
{
    this->add_edge(*producer, *consumer, producer_output_port_id, consumer_input_port_id, edge_type);
}

void Graph::add_edge(const Node &producer, const Node &consumer, EdgeType edge_type)
{
    PortId producer_output_port_id = 0;  // default to zero
    PortId consumer_input_port_id = 0;
    NodeId consumer_node_id = consumer.id();
    const std::unordered_set<Edge> &operand_edges = operands_map_.at(consumer_node_id);

    for (const Edge &operand_edge : operand_edges)
    {
        consumer_input_port_id = std::max(consumer_input_port_id, operand_edge.consumer_input_port_id + 1);
    }
    add_edge(producer, consumer, producer_output_port_id, consumer_input_port_id, edge_type);
}
void Graph::add_edge(const Node *producer, const Node *consumer, EdgeType edge_type)
{
    add_edge(*producer, *consumer, edge_type);
}
void Graph::add_edge(
    Node *producer, Node *consumer, PortId producer_output_port_id, PortId consumer_input_port_id, EdgeType edge_type)
{
    add_edge(*producer, *consumer, producer_output_port_id, consumer_input_port_id, edge_type);
}

std::shared_ptr<EdgeAttributes> Graph::remove_edge(const Edge &edge)
{
    auto attr = get_edge_attributes(edge);
    std::vector<Edge> operand_edges_to_remove;
    for (auto &operand_edge : this->operands_map_[edge.consumer_node_id])
    {
        if (operand_edge.unique_id() == edge.unique_id())
        {
            this->users_map_[edge.producer_node_id].erase(operand_edge);
            operand_edges_to_remove.push_back(operand_edge);
        }
    }
    for (auto &operand_edge : operand_edges_to_remove)
    {
        this->operands_map_[edge.consumer_node_id].erase(operand_edge);
    }

    TT_DBG_ASSERT(operand_edges_to_remove.size(), "Edge not found in graph");

    return attr;
}

std::vector<Node *> Graph::nodes(std::function<bool(Node *)> node_filter) const
{
    std::vector<Node *> ret;
    for (Node *node : nodes_)
        if (node_filter(node) && is_node_visible(node))
            ret.push_back(node);
    return ret;
}

std::vector<Node *> Graph::nodes_by_type(NodeType type) const
{
    return this->nodes([type](Node *n) { return n->node_type() == type; });
}

std::vector<Node *> Graph::nodes_by_subgraph(unsigned int subgraph_id) const
{
    std::vector<Node *> ret;
    for (Node *node : nodes_)
        if (node_id_to_subgraph_id_.at(node->id()) == subgraph_id)
            ret.push_back(node);
    return ret;
}

void Graph::move_node_to_subgraph(NodeId node_id, unsigned int subgraph_id)
{
    node_id_to_subgraph_id_[node_id] = subgraph_id;
    if (subgraph_id >= num_subgraphs_)
        num_subgraphs_ = subgraph_id + 1;
}
unsigned int Graph::get_subgraph_id_for_node(NodeId node_id) const { return node_id_to_subgraph_id_.at(node_id); }

const std::vector<Node *> &Graph::nodes() const { return nodes_; }
int Graph::num_users(const Node *node) const { return this->users_map().at(node->id()).size(); }

void Graph::update_node_name(Node *node, const std::string &new_name)
{
    NodeId node_id = node_name_to_node_id_[node->name()];
    node_name_to_node_id_.erase(node->name());
    node->set_name(new_name);
    node_name_to_node_id_[new_name] = node_id;
}

void Graph::register_module_inputs(const std::vector<NodeId> &module_inputs, bool append)
{
    if (!append)
    {
        this->ordered_module_input_node_ids_.clear();
    }
    for (NodeId module_input : module_inputs)
    {
        this->ordered_module_input_node_ids_.push_back(module_input);
    }
}

std::size_t Graph::remove_module_input(NodeId input)
{
    auto it = std::find(ordered_module_input_node_ids_.begin(), ordered_module_input_node_ids_.end(), input);
    TT_ASSERT(it != ordered_module_input_node_ids_.end());
    std::size_t index = it - ordered_module_input_node_ids_.begin();
    ordered_module_input_node_ids_.erase(it);
    return index;
}

std::size_t Graph::remove_module_output(NodeId output)
{
    auto it = std::find(ordered_module_output_node_ids_.begin(), ordered_module_output_node_ids_.end(), output);
    TT_ASSERT(it != ordered_module_output_node_ids_.end());
    std::size_t index = it - ordered_module_output_node_ids_.begin();
    ordered_module_output_node_ids_.erase(it);
    return index;
}

std::size_t Graph::remove_module_target(NodeId target)
{
    auto it = std::find(ordered_module_target_node_ids_.begin(), ordered_module_target_node_ids_.end(), target);
    TT_ASSERT(it != ordered_module_target_node_ids_.end());
    std::size_t index = it - ordered_module_target_node_ids_.begin();
    ordered_module_target_node_ids_.erase(it);
    return index;
}

void Graph::add_module_input(NodeId input) { ordered_module_input_node_ids_.push_back(input); }

void Graph::add_module_output(NodeId output) { ordered_module_output_node_ids_.push_back(output); }

void Graph::add_module_target(NodeId target) { ordered_module_target_node_ids_.push_back(target); }

void Graph::copy_module_inputs(Graph *old_graph, const std::unordered_map<Node *, Node *> &old_to_new)
{
    this->ordered_module_input_node_ids_.clear();

    for (NodeId id : old_graph->ordered_module_input_node_ids_)
    {
        try
        {
            this->ordered_module_input_node_ids_.push_back(old_to_new.at(old_graph->node_by_id(id))->id());
        }
        catch (std::out_of_range &e)
        {
            log_fatal("Node not mapped to new graph during lowering: {}", old_graph->node_by_id(id)->name());
        }
    }
}

void Graph::register_module_targets(const std::vector<NodeId> &module_targets)
{
    this->ordered_module_target_node_ids_ = module_targets;
}

void Graph::copy_module_targets(Graph *old_graph, const std::unordered_map<Node *, Node *> &old_to_new)
{
    this->ordered_module_target_node_ids_.clear();

    for (NodeId id : old_graph->ordered_module_target_node_ids_)
    {
        try
        {
            this->ordered_module_target_node_ids_.push_back(old_to_new.at(old_graph->node_by_id(id))->id());
        }
        catch (std::out_of_range &e)
        {
            log_fatal("Node not mapped to new graph during lowering: {}", old_graph->node_by_id(id)->name());
        }
    }
}

void Graph::register_module_outputs(const std::vector<NodeId> &module_outputs, bool append)
{
    if (!append)
    {
        this->ordered_module_output_node_ids_.clear();
    }
    for (NodeId module_output : module_outputs)
    {
        this->ordered_module_output_node_ids_.push_back(module_output);
    }
}

void Graph::copy_module_outputs(Graph *old_graph, const std::unordered_map<Node *, Node *> &old_to_new)
{
    this->ordered_module_output_node_ids_.clear();
    for (NodeId id : old_graph->ordered_module_output_node_ids_)
    {
        try
        {
            this->ordered_module_output_node_ids_.push_back(old_to_new.at(old_graph->node_by_id(id))->id());
        }
        catch (std::out_of_range &e)
        {
            log_fatal("Node not mapped to new graph during lowering: {}", old_graph->node_by_id(id)->name());
        }
    }
}

std::ostream &operator<<(std::ostream &out, const Edge &e)
{
    out << "(" << e.producer_node_id << "@" << e.producer_output_port_id << " -> " << e.consumer_node_id << "@"
        << e.consumer_input_port_id << ")";
    return out;
}
std::ostream &operator<<(std::ostream &out, const Graph &g)
{
    out << "Nodes:\n";
    for (const auto &[node_id, node] : g.nodes_map())
    {
        out << "\t" << *node << ": " << node->name() << " " << node->shape();
        out << "\n";
    }

    out << "ordered node operands:\n";
    for (const auto &[node_id, node] : g.nodes_map())
    {
        auto edges = g.operand_edges(node);
        out << "\t" << node_id << ": {";
        for (const auto &edge : edges)
        {
            out << edge << ", ";
        }
        out << "}\n";
    }

    out << "ordered node users:\n";
    for (const auto &[node_id, node] : g.nodes_map())
    {
        auto edges = g.user_edges(node);
        out << "\t" << node_id << ": {";
        for (const auto &edge : edges)
        {
            out << edge << ", ";
        }
        out << "}\n";
    }

    return out;
}

// Return inputs to the graph in way module/user expects
std::vector<Node *> Graph::ordered_module_inputs() const
{
    std::vector<Node *> ordered_inputs;
    for (auto input_node_id : this->ordered_module_input_node_ids_)
    {
        Node *node = this->node_by_id(input_node_id);

        if (this->is_node_visible(node) && this->user_edges(node).size() != 0)
        {
            ordered_inputs.push_back(this->node_by_id(input_node_id));
        }
    }
    return ordered_inputs;
}

std::vector<unsigned int> Graph::get_ordered_input_subgraph_indices() const
{
    std::vector<unsigned int> ordered_input_subgraph_indices;
    for (auto input_node_id : this->ordered_module_input_node_ids_)
    {
        ordered_input_subgraph_indices.push_back(node_id_to_subgraph_id_.at(input_node_id));
    }
    return ordered_input_subgraph_indices;
}

std::vector<std::string> Graph::get_ordered_input_names() const
{
    std::vector<std::string> ordered_inputs;
    for (auto input_node_id : this->ordered_module_input_node_ids_)
    {
        ordered_inputs.push_back(this->node_by_id(input_node_id)->name());
    }
    return ordered_inputs;
}

std::vector<std::string> Graph::get_ordered_target_names() const
{
    std::vector<std::string> ordered_inputs;
    for (auto input_node_id : this->ordered_module_target_node_ids_)
    {
        ordered_inputs.push_back(this->node_by_id(input_node_id)->name());
    }
    return ordered_inputs;
}

std::vector<std::string> Graph::get_ordered_input_gradient_names() const
{
    std::vector<std::string> input_grads;
    input_grads.reserve(this->ordered_module_input_node_ids_.size());
    for (auto input_node_id : this->ordered_module_input_node_ids_)
    {
        Node *node = node_by_id(input_node_id);
        if (node->as<InputNode>()->is_gradient())
        {
            input_grads.push_back(node->name());
        }
    }
    return input_grads;
}

std::vector<Node *> Graph::ordered_intermediates() const
{
    std::vector<Node *> ordered_intermediates;
    for (auto node : this->nodes())
    {
        if (node->node_type() == NodeType::kOutput && node->as<OutputNode>()->is_intermediate())
        {
            ordered_intermediates.push_back(node);
        }
    }

    return ordered_intermediates;
}

std::vector<std::string> Graph::get_ordered_intermediate_names() const
{
    std::vector<std::string> ordered_intermediate_names;
    for (auto node : this->nodes())
    {
        if (node->node_type() == NodeType::kOutput && node->as<OutputNode>()->is_intermediate())
        {
            ordered_intermediate_names.push_back(node->name());
        }
    }

    return ordered_intermediate_names;
}

std::vector<bool> Graph::get_ordered_input_requires_grad() const
{
    std::vector<bool> requires_grad;
    for (auto input_node_id : this->ordered_module_input_node_ids_)
    {
        Node *node = node_by_id(input_node_id);
        requires_grad.push_back(node->as<graphlib::InputNode>()->requires_grad());
    }
    return requires_grad;
}

std::vector<bool> Graph::get_ordered_output_requires_grad() const
{
    std::vector<bool> requires_grad;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        Node *node = node_by_id(output_node_id);
        requires_grad.push_back(node->as<graphlib::OutputNode>()->requires_grad());
    }
    return requires_grad;
}

std::vector<std::vector<std::uint32_t>> Graph::get_ordered_input_shapes() const
{
    std::vector<std::vector<std::uint32_t>> ordered_inputs;
    for (auto input_node_id : this->ordered_module_input_node_ids_)
    {
        ordered_inputs.push_back(this->node_by_id(input_node_id)->shape().as_vector());
    }
    return ordered_inputs;
}
std::vector<std::vector<std::uint32_t>> Graph::get_ordered_output_shapes() const
{
    std::vector<std::vector<std::uint32_t>> ordered_outputs;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        ordered_outputs.push_back(this->node_by_id(output_node_id)->shape().as_vector());
    }
    return ordered_outputs;
}
std::vector<std::vector<std::uint32_t>> Graph::get_ordered_intermediate_shapes() const
{
    std::vector<std::vector<std::uint32_t>> ordered_intermediates;
    for (const auto &op_name : this->get_ordered_intermediate_names())
    {
        ordered_intermediates.push_back(this->get_node_by_name(op_name)->shape().as_vector());
    }
    return ordered_intermediates;
}
std::vector<std::vector<std::uint32_t>> Graph::get_ordered_target_shapes() const
{
    std::vector<std::vector<std::uint32_t>> ordered_targets;
    for (auto target_node_id : this->ordered_module_target_node_ids_)
    {
        ordered_targets.push_back(this->node_by_id(target_node_id)->shape().as_vector());
    }
    return ordered_targets;
}
std::vector<unsigned int> Graph::get_ordered_target_subgraph_indices() const
{
    std::vector<unsigned int> ordered_target_subgraph_indices;
    for (auto target_node_id : this->ordered_module_target_node_ids_)
    {
        ordered_target_subgraph_indices.push_back(node_id_to_subgraph_id_.at(target_node_id));
    }
    return ordered_target_subgraph_indices;
}

std::vector<std::string> Graph::get_ordered_output_gradient_names() const
{
    std::vector<std::string> ordered_inputs;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        Node *node = node_by_id(output_node_id);
        std::vector<Edge> losses =
            user_edges(node, [](Edge edge) { return edge.edge_type == EdgeType::kAutogradOutputToLoss; });
        TT_ASSERT(losses.size() <= 1, "Each output should have at most one incoming loss");
        if (losses.size() > 0)
            ordered_inputs.push_back(node_by_id(losses[0].consumer_node_id)->name());
    }

    return ordered_inputs;
}

std::vector<std::vector<int>> Graph::get_ordered_input_tile_dims() const
{
    std::vector<std::vector<int>> ordered_input_tile_dims;
    for (auto input_node_id : this->ordered_module_input_node_ids_)
    {
        auto tile_dim = this->node_by_id(input_node_id)->shape().get_tile_dim();
        std::vector<int> tile_dims{
            get_row_size_from_tile_size(tile_dim),
            get_col_size_from_tile_size(tile_dim),
        };
        ordered_input_tile_dims.push_back(tile_dims);
    }
    return ordered_input_tile_dims;
}

std::vector<std::vector<int>> Graph::get_ordered_parameter_tile_dims() const
{
    std::vector<std::vector<int>> ordered_param_tile_dims;
    for (auto param_node : this->get_parameter_nodes())
    {
        auto tile_dim = param_node->shape().get_tile_dim();
        std::vector<int> tile_dims{
            get_row_size_from_tile_size(tile_dim),
            get_col_size_from_tile_size(tile_dim),
        };
        ordered_param_tile_dims.push_back(tile_dims);
    }
    return ordered_param_tile_dims;
}

std::vector<std::vector<int>> Graph::get_ordered_constant_tile_dims() const
{
    std::vector<std::vector<int>> ordered_const_tile_dims;
    for (auto const_node : this->get_constant_nodes())
    {
        auto tile_dim = const_node->shape().get_tile_dim();
        std::vector<int> tile_dims{
            get_row_size_from_tile_size(tile_dim),
            get_col_size_from_tile_size(tile_dim),
        };
        ordered_const_tile_dims.push_back(tile_dims);
    }
    return ordered_const_tile_dims;
}

// Return outputs to the graph in way module/user expects
std::vector<Node *> Graph::ordered_module_outputs() const
{
    std::vector<Node *> ordered_outputs;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        ordered_outputs.push_back(this->node_by_id(output_node_id));
    }
    return ordered_outputs;
}

std::vector<unsigned int> Graph::get_ordered_output_subgraph_indices() const
{
    std::vector<unsigned int> ordered_output_subgraph_indices;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        ordered_output_subgraph_indices.push_back(node_id_to_subgraph_id_.at(output_node_id));
    }
    return ordered_output_subgraph_indices;
}

std::vector<Node *> Graph::ordered_module_outputs_by_subgraph_index(unsigned int subgraph_index) const
{
    std::vector<Node *> ordered_outputs;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        if (node_id_to_subgraph_id_.at(output_node_id) == subgraph_index)
            ordered_outputs.push_back(this->node_by_id(output_node_id));
    }
    return ordered_outputs;
}

std::vector<Node *> Graph::ordered_partial_datacopy_outputs() const
{
    return sorted(
        this,
        nodes(),
        [this](Node *output)
        {
            if (output->node_type() != NodeType::kOutput)
                return false;
            auto users = this->user_edges(output);
            return users.size() == 1 and users.front().edge_type == EdgeType::kPartialDataCopy;
        });
}

std::vector<Node *> Graph::get_constant_nodes(bool recurse) const
{
    std::vector<Node *> constants;
    for (Node *node : nodes_)
    {
        if (node->node_type() == NodeType::kInput)
        {
            InputNode *input_node = node->as<InputNode>();
            if (auto const_eval_graph = input_node->get_consteval_graph(); const_eval_graph != nullptr and recurse)
            {
                std::vector<Node *> subgraph_constant_nodes = const_eval_graph->get_graph()->get_constant_nodes();
                constants.insert(constants.end(), subgraph_constant_nodes.begin(), subgraph_constant_nodes.end());
            }
            if (input_node->is_constant())
            {
                constants.push_back(node);
            }
        }
    }
    return constants;
}
std::vector<std::string> Graph::get_constant_names() const
{
    std::vector<std::string> constant_names;
    for (Node *constant_node : this->get_constant_nodes())
    {
        constant_names.push_back(constant_node->name());
    }
    return constant_names;
}

std::vector<Node *> Graph::get_optimizer_parameter_nodes() const
{
    std::vector<Node *> parameters;
    for (Node *node : nodes_)
    {
        if (node->node_type() == NodeType::kInput)
        {
            InputNode *input_node = node->as<InputNode>();
            if (input_node->is_optimizer_parameter())
            {
                parameters.push_back(node);
            }
        }
    }
    return parameters;
}

std::vector<Node *> Graph::get_parameter_nodes() const
{
    std::vector<Node *> parameters;
    for (Node *node : nodes_)
    {
        if (node->node_type() == NodeType::kInput)
        {
            InputNode *input_node = node->as<InputNode>();
            if (input_node->is_parameter())
            {
                parameters.push_back(node);
            }
        }
    }
    return parameters;
}

std::vector<std::string> Graph::get_ordered_output_names() const
{
    std::vector<std::string> ordered_outputs;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        ordered_outputs.push_back(this->node_by_id(output_node_id)->name());
    }
    return ordered_outputs;
}

std::vector<std::string> Graph::get_ordered_external_output_names() const
{
    std::vector<std::string> ordered_outputs;
    for (auto output_node_id : this->ordered_module_output_node_ids_)
    {
        if (this->node_by_id(output_node_id)->as<OutputNode>()->output_type() == OutputType::External)
        {
            ordered_outputs.push_back(this->node_by_id(output_node_id)->name());
        }
    }
    return ordered_outputs;
}

bool Graph::contains_nodes_of_epoch_type(NodeEpochType node_epoch_type) const
{
    // Cache if it starts getting slow?
    for (Node *node : nodes_)
    {
        if (node->get_epoch_type() == node_epoch_type)
            return true;
    }

    return false;
}

// Autograd mapping retrieval
bool Graph::contains_bwd_nodes() const { return this->contains_nodes_of_epoch_type(NodeEpochType::Backward); }

// Autograd mapping retrieval
bool Graph::contains_opt_nodes() const { return this->contains_nodes_of_epoch_type(NodeEpochType::Optimizer); }

bool Graph::contains_recompute_nodes() const
{
    for (const auto &[node_id, users_edge_set] : this->users_map_)
    {
        for (const auto &edge : users_edge_set)
        {
            if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute)
            {
                return true;
            }
        }
    }
    return false;
}

// For given input index, return list of dims that need to be broadcast within a tile
std::vector<int> Graph::get_tile_broadcast_dims_for_input(std::uint32_t input_index) const
{
    TT_ASSERT(input_index < this->ordered_module_input_node_ids_.size());
    InputNode *input = node_by_id(this->ordered_module_input_node_ids_[input_index])->as<InputNode>();
    return input->get_tile_broadcast_dims();
}

std::vector<int> Graph::get_tile_broadcast_dims_for_bw_input(std::uint32_t output_index) const
{
    std::vector<std::string> gradient_names = get_ordered_output_gradient_names();
    TT_ASSERT(output_index < gradient_names.size());
    InputNode *input = get_node_by_name(gradient_names[output_index])->as<InputNode>();
    return input->get_tile_broadcast_dims();
}

std::vector<int> Graph::get_tile_broadcast_dims_for_target(std::uint32_t target_index) const
{
    TT_ASSERT(target_index < this->ordered_module_target_node_ids_.size());
    InputNode *input = node_by_id(this->ordered_module_target_node_ids_[target_index])->as<InputNode>();
    return input->get_tile_broadcast_dims();
}

std::unordered_map<int, std::vector<Node *>> Graph::get_recompute_nodes(Node *fwd_node) const
{
    std::unordered_map<int, std::vector<Node *>> ret;
    for (Edge edge : user_edges(fwd_node))
    {
        if (edge.edge_type == EdgeType::kAutogradFwdToRecompute)
        {
            ret[edge.producer_output_port_id].push_back(node_by_id(edge.consumer_node_id));
        }
    }
    return ret;
}

std::unordered_map<int, std::vector<Node *>> Graph::get_gradient_nodes(Node *fwd_node) const
{
    std::unordered_map<int, std::vector<Node *>> ret;
    for (Edge edge : user_edges(fwd_node))
    {
        if (edge.edge_type == EdgeType::kAutogradFwdToGradient)
        {
            Node *gradient = node_by_id(edge.consumer_node_id);
            Node *gradient_op =
                (gradient->node_type() == NodeType::kQueue) ? this->data_operands(gradient).at(0) : gradient;

            ret[edge.producer_output_port_id].push_back(gradient_op);
        }
    }
    return ret;
}

std::unordered_map<int, std::vector<Node *>> Graph::get_bwd_nodes(Node *fwd_node) const
{
    std::unordered_map<int, std::vector<Node *>> ret;
    for (Edge edge : user_edges(fwd_node))
    {
        if (edge.edge_type == EdgeType::kAutogradFwdToBwd)
        {
            Node *bwd_node = this->node_by_id(edge.consumer_node_id);

            // if bwd_node has incoming gradient-edges, or is bwd-node, skip
            bool skip = false;
            OpNode *op_node = dynamic_cast<OpNode *>(bwd_node);
            if (op_node and op_node->is_gradient_op())
            {
                skip = true;
            }

            for (const Edge &op_edge : this->operand_edges(bwd_node))
            {
                if (op_edge.edge_type == EdgeType::kAutogradFwdToGradient)
                {
                    Node *producer = node_by_id(op_edge.producer_node_id);
                    if (op_edge.producer_node_id != fwd_node->id())
                    {
                        skip = producer->node_type() != NodeType::kInput;
                    }
                }
            }
            if (not skip)
            {
                ret[edge.producer_output_port_id].push_back(node_by_id(edge.consumer_node_id));
            }
        }
    }
    return ret;
}

std::unordered_map<int, std::vector<Node *>> Graph::get_opt_nodes(Node *fwd_node) const
{
    std::unordered_map<int, std::vector<Node *>> ret;
    for (Edge edge : user_edges(fwd_node))
    {
        if (edge.edge_type == EdgeType::kAutogradFwdToOptimizer)
        {
            ret[edge.producer_output_port_id].push_back(node_by_id(edge.consumer_node_id));
        }
    }
    return ret;
}

void Graph::dump(std::string const &pass_name) const { reportify::dump_graph(name(), pass_name, this); }

}  // namespace graphlib
}  // namespace tt
