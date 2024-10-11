// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "autograd/autograd.hpp"

#include "autograd/binding.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;

namespace tt
{

namespace autograd
{

autograd_engine::autograd_engine(Graph *graph, autograd_config config) : graph(graph), config(config) {}

void set_requires_grad_on_outputs(grad_map &requires_grad_map, Graph *graph, Node *node, bool requires_grad)
{
    for (auto edge : graph->user_data_edges(node))
    {
        requires_grad_map.insert(std::make_pair(edge.unique_id(), requires_grad));
    }
}

// Propagate requires_grad from inputs to all edges of the graph, creating an edge->bool map
grad_map autograd_engine::propagate_requires_grad()
{
    auto topo_order = graphlib::topological_sort(*graph);

    grad_map requires_grad_map;
    for (Node *node : topo_order)
    {
        if (node->node_type() == NodeType::kInput)
        {
            // Inputs get requires_grad from the node
            bool requires_grad = node->as<graphlib::InputNode>()->requires_grad();
            set_requires_grad_on_outputs(requires_grad_map, graph, node, requires_grad);
        }
        else if (node->node_type() != NodeType::kOutput)
        {
            // Output edge of the node needs requires_grad if any of the input edges need it
            bool requires_grad = false;
            for (auto edge : graph->operand_data_edges(node))
            {
                try
                {
                    if (requires_grad_map.at(edge.unique_id()))
                    {
                        requires_grad = true;
                        break;
                    }
                }
                catch (std::out_of_range &e)
                {
                    throw std::runtime_error(
                        "requires_grad missing for edge on " + node->name() + ", something went wrong.");
                }
            }
            set_requires_grad_on_outputs(requires_grad_map, graph, node, requires_grad);
        }
    }

    return requires_grad_map;
}

// Register fwd->bwd and bwd->fwd relationship
void autograd_engine::add_fwd_to_bwd_map(Node *fwd, Node *bwd, int operand_index, bool out_gradient)
{
    log_debug(tt::LogAutograd, "add_fwd_to_bwd: {} (operand: {}) -> {}", fwd->name(), operand_index, bwd->name());

    graph->add_edge(Edge(fwd->id(), operand_index, bwd->id(), 0, graphlib::EdgeType::kAutogradFwdToBwd));
    TT_ASSERT(fwd->is_forward());
    TT_ASSERT(bwd->is_backward());

    if (out_gradient)
    {
        add_fwd_to_out_gradient_map(fwd, bwd);
    }
}

void autograd_engine::add_fwd_to_optimizer_edge(Node *fwd, Node *opt, int operand_index)
{
    log_debug(tt::LogAutograd, "add_fwd_to_opt: {} (operand: {}) -> {}", fwd->name(), operand_index, opt->name());

    graph->add_edge(Edge(fwd->id(), operand_index, opt->id(), 0, graphlib::EdgeType::kAutogradFwdToOptimizer));
    TT_ASSERT(fwd->is_forward());
    TT_ASSERT(opt->is_optimizer());
}

// Register fwd->out_gradient
void autograd_engine::add_fwd_to_out_gradient_map(Node *fwd, Node *out_gradient)
{
    log_debug(tt::LogAutograd, "add_fwd_to_out_gradient_map: {} -> {} ", fwd->name(), out_gradient->name());
    fwd_to_out_gradient_map[fwd].push_back(out_gradient);
    graph->add_edge(fwd, out_gradient, graphlib::EdgeType::kAutogradFwdToGradient);

    TT_ASSERT(fwd->is_forward());
    TT_ASSERT(out_gradient->is_backward());
}

// Combine incoming gradients by adding them, and return the new combined node
Node *autograd_engine::combine_incoming_gradients(Node *node)
{
    auto it = fwd_to_out_gradient_map.find(node);
    if (it == fwd_to_out_gradient_map.end())
        return nullptr;  // no gradients, requires_grad must've been false on this branch

    std::vector<Node *> out_grads = fwd_to_out_gradient_map.at(node);
    TT_ASSERT(out_grads.size() > 0);

    if (out_grads.size() == 1)
    {
        return out_grads[0];  // nothing to combine
    }

    NodeContext sum(out_grads[0]);
    for (int i = 1; i < (int)out_grads.size(); i++)
    {
        NodeContext out_grad(out_grads[i]);
        sum = create_op(graphlib::OpType("add"), {sum, out_grad}, node, 0, i - 1, "combine");
    }

    Node *final_out = graph->node_by_id(sum.id);

    // Remove edges, add a combined one
    for (Edge edge :
         graph->user_edges(node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kAutogradFwdToGradient; }))
    {
        graph->remove_edge(edge);
    }
    graph->add_edge(node, final_out, graphlib::EdgeType::kAutogradFwdToGradient);
    fwd_to_out_gradient_map.at(node).clear();
    fwd_to_out_gradient_map.at(node).push_back(final_out);
    return final_out;
}

bool is_fwd_to_gradient_edge(const Edge &edge) { return edge.edge_type == graphlib::EdgeType::kAutogradFwdToGradient; }

// Create backward instructions, and hook them up accordingly
void autograd_engine::create_backward_graph(const grad_map &requires_grad_map)
{
    // Create input nodes for each of the loss tensors, or loop back loss outputs
    std::vector<Node *> nodes = graph->nodes();
    for (Node *node : nodes)
    {
        if (node->node_type() == NodeType::kOutput)
        {
            TT_ASSERT(graph->operand_data_edges(node).size() == 1);
            if (!requires_grad_map.at((graph->operand_data_edges(node)[0]).unique_id()))
                continue;

            graphlib::OutputNode *output = node->as<graphlib::OutputNode>();
            if (output->is_loss_output())
            {
                // Grad of loss is 1. Create constant and use that as "input".
                py::object eval_module = py::module_::import("forge.op.eval");
                auto const_tensor = make_shared_py_object(eval_module.attr("create_constant_tensor_from_tensor")(
                    std::vector<float>{1.0}, node->shape().as_vector(), false, node->output_df()));
                auto const_node = graph->add_node(
                    graphlib::create_node<graphlib::ConstantInputNode>(
                        "loss_grad_constant_" + node->name(), const_tensor, node->shape()),
                    graph->get_subgraph_id_for_node(node->id()));
                const_node->set_backward();
                const_node->set_output_df(node->output_df());

                graphlib::RuntimeTensorTransform runtime_tensor_transform = output->get_runtime_tensor_transform();
                const_node->set_runtime_tensor_transform(runtime_tensor_transform);

                add_fwd_to_bwd_map(graph->data_operands(node)[0], const_node, 0, true);
            }

            else
            {
                // Normal outputs, which will require a loss input to bring its gradient in
                auto input_node = graph->add_node(
                    graphlib::create_node<graphlib::InputNode>(
                        "loss_" + node->name(), graphlib::InputNodeType::Loss, false),
                    graph->get_subgraph_id_for_node(node->id()));

                input_node->set_shape(node->shape());
                input_node->set_backward();
                input_node->set_output_df(node->output_df());
                graphlib::RuntimeTensorTransform runtime_tensor_transform = output->get_runtime_tensor_transform();
                runtime_tensor_transform.swap_original_and_reinterpreted_shapes();
                input_node->set_runtime_tensor_transform(runtime_tensor_transform);

                graph->add_edge(node, input_node, graphlib::EdgeType::kAutogradOutputToLoss);
                add_fwd_to_bwd_map(graph->data_operands(node)[0], input_node, 0, true);

                graph->register_module_inputs({input_node->id()}, true /* append */);
            }
        }
    }

    std::vector<Node *> topo_order =
        tt::graphlib::topological_sort(*graph, [](Node *node) { return node->is_forward(); });

    for (int node_index = topo_order.size() - 1; node_index >= 0; node_index--)
    {
        Node *node = topo_order[node_index];

        if (node->node_type() == NodeType::kOutput)
            continue;

        if (node->node_type() == NodeType::kInput)
        {
            if (!node->as<graphlib::InputNode>()->requires_grad())
                continue;
        }

        // Combine incoming gradients into one, if there's more than one
        Node *out_grad = combine_incoming_gradients(node);

        if (out_grad == nullptr)
        {
            // No gradients for this node, move on
            continue;
        }

        // If broadcast is needed, we have to insert a nop to explicitly broadcast the shape
        std::vector<DimBroadcast> brcst = out_grad->shape().broadcast_dims(node->shape());

        if (brcst.size() != 0)
        {
            Node *nop_node = graph->add_node(
                graphlib::create_node<graphlib::PyOpNode>("broadcast_out_" + node->name(), "nop"),
                graph->get_subgraph_id_for_node(node->id()));
            nop_node->set_shape(node->shape());
            nop_node->set_epoch_type(graphlib::NodeEpochType::Backward);
            nop_node->set_output_df(nop_node->output_df());

            Edge nop_edge(out_grad->id(), 0, nop_node->id(), 0, graphlib::EdgeType::kData);
            graph->add_edge(nop_edge);

            for (DimBroadcast b : brcst)
            {
                TT_ASSERT(std::get<0>(b) == 0, "Only one operand available");
                // Autograd must use explicit bcasts to support ops like matmuls
                constexpr bool explicit_bcast = true;
                int negative_index_bcast_dim = std::get<1>(b) - out_grad->shape().size();
                graph->get_edge_attributes(nop_edge)->set_broadcast_dim(
                    negative_index_bcast_dim, std::get<2>(b), explicit_bcast);
            }

            // Remove edge, replace
            for (Edge edge : graph->user_edges(
                     node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kAutogradFwdToGradient; }))
            {
                graph->remove_edge(edge);
            }
            graph->add_edge(node, nop_node, graphlib::EdgeType::kAutogradFwdToGradient);
            graph->add_edge(node, nop_node, graphlib::EdgeType::kAutogradFwdToBwd);
            fwd_to_out_gradient_map.at(node).clear();
            fwd_to_out_gradient_map.at(node).push_back(nop_node);

            out_grad = nop_node;
        }

        if (node->node_type() == NodeType::kInput)
        {
            if (node->as<graphlib::InputNode>()->is_parameter())
            {
                log_debug(tt::LogAutograd, "Setting gradient op flag on {}", out_grad->name());
                out_grad->as<graphlib::PyOpNode>()->set_gradient_op();  // this is a gradient accumulator

                // Create gradient queue to write gradient accumulation to in prologue
                auto epilogue_q = graph->add_node(
                    graphlib::create_node<graphlib::QueueNode>(
                        "grad_acc_" + node->name(), graphlib::QueueNodeType::GradAccumulator),
                    graph->get_subgraph_id_for_node(node->id()));
                epilogue_q->set_backward();
                epilogue_q->set_shape(node->shape());
                epilogue_q->set_output_df(out_grad->output_df());

                graph->add_edge(out_grad, epilogue_q);

                // Move fwd->gradient edge from the op to the queue, so that optimizer uses queue as the input
                std::vector<Edge> old_edges = graph->operand_edges(out_grad, is_fwd_to_gradient_edge);
                TT_ASSERT(old_edges.size() == 1, "Gradient op has no FwdToGradient incoming edge");
                graph->remove_edge(old_edges[0]);
                graph->add_edge(
                    graph->node_by_id(old_edges[0].producer_node_id),
                    epilogue_q,
                    old_edges[0].producer_output_port_id,
                    graphlib::PortId(0),
                    graphlib::EdgeType::kAutogradFwdToGradient);
            }
            else if (node->as<graphlib::InputNode>()->is_activation())
            {
                // if this an input into the graph, then we need to output the gradients
                auto output_node = graph->add_node(
                    graphlib::create_node<graphlib::OutputNode>("output_grad_" + node->name()),
                    graph->get_subgraph_id_for_node(node->id()));
                output_node->set_backward();
                output_node->set_shape(node->shape());
                output_node->set_output_df(node->output_df());

                graph->add_edge(node, output_node, graphlib::EdgeType::kAutogradInputToGradientOut);
                graph->add_edge(out_grad, output_node);
            }

            if (graphlib::ConstEvalGraph *consteval_graph = node->as<graphlib::InputNode>()->get_consteval_graph())
            {
                consteval_graph->set_needs_autograd(true);
                consteval_graph->autograd();
            }

            continue;
        }

        if (node->node_type() != NodeType::kPyOp)
            continue;

        graphlib::PyOpNode *op_node = node->as<graphlib::PyOpNode>();

        // Not input or outputs -- differentiate
        for (Edge edge : graph->operand_data_edges(node))
        {
            try
            {
                if (!requires_grad_map.at(edge.unique_id()))
                    continue;
            }
            catch (std::out_of_range &e)
            {
                throw std::runtime_error(
                    "requires_grad missing for edge on " + node->name() + ", something went wrong.");
            }

            log_debug(
                tt::LogAutograd,
                "Edge from {} to {} requires grad.",
                graph->node_by_id(edge.producer_node_id)->name(),
                node->name());

            std::vector<NodeContext> operands;
            for (Node *operand : graph->data_operands(node))
            {
                operands.push_back(NodeContext(operand));
                operands.back().shape = node->shape_of_operand(graph, operand);  // Expand out bcast shapes
            }
            NodeContext gradient(out_grad);

            NodeContext ret_gradient = insert_backward(
                {this, node, (int)edge.consumer_input_port_id},
                op_node->op_type(),
                edge.consumer_input_port_id,
                operands,
                NodeContext(node),
                gradient);

            // Check for broadcast, and create sequence of reduce ops if there are any
            NodeContext last_out = ret_gradient;
            int created_op_index = 0;
            for (graphlib::OpType tm : graph->get_edge_attributes(edge)->get_tms())
            {
                if (tm.op == "broadcast")
                {
                    TT_ASSERT(tm.attr.size() <= 3);
                    log_debug(
                        tt::LogAutograd,
                        "Edge has broadcast: dim={} size={}",
                        std::get<int>(tm.attr[0]),
                        std::get<int>(tm.attr[1]));
                    int dim = std::get<int>(tm.attr[0]);

                    NodeContext src = last_out;
                    last_out = create_op(
                        OpType("reduce_sum", {dim}),
                        {src},
                        node,
                        edge.consumer_input_port_id,
                        created_op_index++,
                        "brcst",
                        false);
                }
            }

            add_fwd_to_out_gradient_map(graph->node_by_id(edge.producer_node_id), graph->node_by_id(last_out.id));
        }
    }
}

void autograd_engine::create_optimizer_graph()
{
    if (config.optimizer.is_none())
    {
        return;
    }

    // for each parameter with requires_grad, we're going to inject an optimizer graph
    std::unordered_set<Node *> visited;
    std::vector<Node *> topo_order =
        graphlib::topological_sort(*graph);  // copy into vector to avoid iterator invalidation

    for (Node *input_node : topo_order)
    {
        if (visited.find(input_node) != visited.end())
        {
            continue;
        }

        if (input_node->node_type() == NodeType::kInput)
        {
            bool requires_grad = input_node->as<graphlib::InputNode>()->requires_grad();
            bool is_param = input_node->as<graphlib::InputNode>()->is_parameter();

            if (requires_grad && is_param)
            {
                auto gradient_edges = graph->user_edges(input_node, is_fwd_to_gradient_edge);
                auto user_data_edges = graph->user_data_edges(input_node);

                TT_ASSERT(gradient_edges.size() == 1);
                TT_ASSERT(user_data_edges.size() >= 1);

                for (const auto &gradient_edge : gradient_edges)
                {
                    const auto &input_to_fwd_node_edge = user_data_edges.at(0);
                    Node *gradient = graph->node_by_id(gradient_edge.consumer_node_id);

                    auto generate_op_trace_fcn = config.optimizer.attr("generate_op_trace");
                    autograd_context context = {
                        .autograd = this,
                        .current_fwd_op = input_node,
                        .operand = (int)input_to_fwd_node_edge.consumer_input_port_id,
                        .epoch_type = graphlib::NodeEpochType::Optimizer,
                    };
                    NodeContext optimizer_output =
                        generate_op_trace_fcn(context, NodeContext(input_node), NodeContext(gradient))
                            .cast<NodeContext>();

                    graph->add_edge(
                        graph->node_by_id(optimizer_output.id),
                        input_node,
                        graphlib::PortId(0),
                        graphlib::PortId(0),
                        graphlib::EdgeType::kDataLoopback);

                    for (auto outgoing_edge : graph->user_edges(input_node))
                    {
                        graphlib::NodeId outgoing_node_id = outgoing_edge.consumer_node_id;
                        graphlib::Node *outgoing_node = graph->node_by_id(outgoing_node_id);

                        if (outgoing_node->node_type() == NodeType::kInput)
                        {
                            bool is_optimizer_param =
                                outgoing_node->as<graphlib::InputNode>()->is_optimizer_parameter();
                            bool has_dataloopback = false;

                            for (auto incoming_edge : graph->operand_edges(outgoing_node))
                            {
                                if (incoming_edge.edge_type == graphlib::EdgeType::kDataLoopback)
                                {
                                    has_dataloopback = true;
                                    break;
                                }
                            }

                            if (is_optimizer_param && has_dataloopback &&
                                !outgoing_node->as<graphlib::TaggedNode>()->has_tag("dont_consteval"))
                            {
                                outgoing_node->as<graphlib::InputNode>()->clone_consteval_graph_from(input_node);
                            }
                        }
                    }
                }

                visited.insert(input_node);
            }
        }
    }
}

Graph *autograd_engine::run()
{
    log_debug(tt::LogAutograd, "Autograd For Graph: {}", graph->name());
    log_debug(tt::LogAutograd, "Propagating requires_grad");
    grad_map requires_grad_map = propagate_requires_grad();
    create_backward_graph(requires_grad_map);
    create_optimizer_graph();

    if (config.recompute)
    {
        insert_recompute_ops();
    }

    return graph;
}

// Create a backward op for the given fwd op's operand
NodeContext autograd_engine::create_op(
    graphlib::OpType type,
    std::vector<NodeContext> operands,
    Node *current_fwd_op,
    int operand_index,
    int created_op_index,
    std::string name_prefix,
    bool copy_golden_transforms)
{
    log_debug(tt::LogAutograd, "create_op for {} operand {}", current_fwd_op->name(), operand_index);
    std::string op_name = "bw_in" + std::to_string(operand_index) + "_" + current_fwd_op->name() + "_";
    if (name_prefix.length() > 0)
        op_name += name_prefix + "_";
    op_name += type.op + "_" + std::to_string(created_op_index);

    auto node = graph->add_node(
        graphlib::create_node<graphlib::PyOpNode>(op_name, type),
        graph->get_subgraph_id_for_node(current_fwd_op->id()));

    int i = 0;
    for (NodeContext &n : operands)
    {
        graph->add_edge(Edge(n.id, n.output_index, node->id(), i++, graphlib::EdgeType::kData));
    }

    graphlib::calculate_and_set_node_shape(graph, node);
    node->set_backward();
    node->set_output_df_from_operands(graph);

    // For certain ops we may want to not copy transforms as they may be invalid. I.e reduce_sums
    // inserted as the derivative for broadcast. In this case we wan't to disable verification on the op too
    if (graphlib::OpNode *fwd_op_node = dynamic_cast<graphlib::OpNode *>(current_fwd_op))
    {
        if (copy_golden_transforms)
            node->set_golden_transforms(fwd_op_node->get_golden_transforms());
        else if (graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(node))
            op_node->disable_golden_id();
    }

    add_fwd_to_bwd_map(current_fwd_op, node, operand_index);
    return NodeContext(node);
}

// Create a backward op for the given fwd op's operand
NodeContext autograd_engine::create_optimizer_op(
    graphlib::OpType type,
    std::vector<NodeContext> operands,
    Node *current_fwd_op,
    int operand_index,
    int created_op_index,
    std::string name_prefix)
{
    std::string op_name = "opt_in" + std::to_string(operand_index) + "_" + current_fwd_op->name() + "_";
    if (name_prefix.length() > 0)
    {
        op_name += name_prefix + "_";
    }

    op_name += type.op + "_" + std::to_string(created_op_index);

    auto node = graph->add_node(
        graphlib::create_node<graphlib::PyOpNode>(op_name, type),
        graph->get_subgraph_id_for_node(current_fwd_op->id()));
    for (size_t i = 0; i < operands.size(); ++i)
    {
        const NodeContext &n = operands[i];
        graph->add_edge(Edge(n.id, n.output_index, node->id(), i, graphlib::EdgeType::kData));
    }

    std::vector<Shape> operand_shapes;
    for (NodeContext &n : operands)
    {
        operand_shapes.push_back(graph->node_by_id(n.id)->shape());
    }
    std::tuple<Shape, std::vector<DimBroadcast>> shape_data = get_op_shape(type, operand_shapes, false);

    node->set_shape(Shape(std::get<0>(shape_data)));
    node->set_optimizer();
    node->set_output_df_from_operands(graph);

    // Set broadcast attributes on edges
    for (Edge &e : graph->operand_data_edges(node))
    {
        for (DimBroadcast &b : std::get<1>(shape_data))
        {
            int operand = std::get<0>(b);
            if (operand == (int)e.consumer_input_port_id)
            {
                int dim = std::get<1>(b);
                int size = std::get<2>(b);
                graph->get_edge_attributes(e)->set_broadcast_dim(dim, size);
            }
        }
    }

    add_fwd_to_optimizer_edge(current_fwd_op, node, operand_index);
    return NodeContext(node);
}

static void tag_disable_consteval(bool disable_consteval, Node *node)
{
    if (disable_consteval)
    {
        node->as<graphlib::TaggedNode>()->tag("dont_consteval", "true");
    }
}

// Create an integer constant used in backward calculations (typically a negative one)
template <>
NodeContext autograd_engine::create_constant(
    Node *current_fwd_op, int operand_index, int value, int created_op_index, graphlib::NodeEpochType epoch_type)
{
    auto node = graph->add_node(
        graphlib::create_node<graphlib::ConstantInputNode>(
            "input_constant_" + current_fwd_op->name() + "_" + std::to_string(created_op_index), value),
        graph->get_subgraph_id_for_node(current_fwd_op->id()));

    node->set_shape(Shape::create({1}));
    node->set_output_df(current_fwd_op->output_df());

    if (epoch_type == graphlib::NodeEpochType::Backward)
    {
        node->set_backward();
        add_fwd_to_bwd_map(current_fwd_op, node, operand_index);
    }
    else if (epoch_type == graphlib::NodeEpochType::Optimizer)
    {
        node->set_optimizer();
        add_fwd_to_optimizer_edge(current_fwd_op, node, operand_index);
    }

    return NodeContext(node);
}
template <>
NodeContext autograd_engine::create_constant(
    Node *current_fwd_op, int operand_index, float value, int created_op_index, graphlib::NodeEpochType epoch_type)
{
    auto node = graph->add_node(
        graphlib::create_node<graphlib::ConstantInputNode>(
            "input_constant_" + current_fwd_op->name() + "_" + std::to_string(created_op_index), value),
        graph->get_subgraph_id_for_node(current_fwd_op->id()));

    node->set_shape(Shape::create({1}));
    node->set_output_df(current_fwd_op->output_df());

    if (epoch_type == graphlib::NodeEpochType::Backward)
    {
        node->set_backward();
        add_fwd_to_bwd_map(current_fwd_op, node, operand_index);
    }
    else if (epoch_type == graphlib::NodeEpochType::Optimizer)
    {
        node->set_optimizer();
        add_fwd_to_optimizer_edge(current_fwd_op, node, operand_index);
    }

    return NodeContext(node);
}

NodeContext autograd_engine::create_constant(
    Node *current_fwd_op,
    int operand_index,
    std::shared_ptr<void> tensor,
    graphlib::Shape shape,
    int created_op_index,
    graphlib::NodeEpochType epoch_type)
{
    auto node = graph->add_node(
        graphlib::create_node<graphlib::ConstantInputNode>(
            "input_constant_" + current_fwd_op->name() + "_" + std::to_string(created_op_index), tensor, shape),
        graph->get_subgraph_id_for_node(current_fwd_op->id()));

    node->set_shape(shape);
    node->set_output_df(current_fwd_op->output_df());

    if (epoch_type == graphlib::NodeEpochType::Backward)
    {
        node->set_backward();
        add_fwd_to_bwd_map(current_fwd_op, node, operand_index);
    }
    else if (epoch_type == graphlib::NodeEpochType::Optimizer)
    {
        node->set_optimizer();
        add_fwd_to_optimizer_edge(current_fwd_op, node, operand_index);
    }

    return NodeContext(node);
}

NodeContext autograd_engine::create_input(
    Node *current_fwd_op,
    int operand_index,
    int created_op_index,
    graphlib::NodeEpochType epoch_type,
    std::string &suffix_identifier,
    std::vector<std::uint32_t> tensor_shape,
    bool copy_consteval_operations,
    bool disable_consteval)
{
    std::string base_string = (epoch_type == graphlib::NodeEpochType::Backward) ? "bwd" : "opt";

    graphlib::InputNode *node = graph->add_node(
        graphlib::create_node<graphlib::InputNode>(
            "input_" + base_string + "_" + current_fwd_op->name() + "_" + std::to_string(created_op_index) + "." +
                suffix_identifier,
            graphlib::InputNodeType::OptimizerParameter,
            false),
        graph->get_subgraph_id_for_node(current_fwd_op->id()));
    if (copy_consteval_operations)
    {
        node->clone_consteval_graph_from(current_fwd_op);
    }

    node->set_shape(Shape::create(tensor_shape));
    node->set_output_df(current_fwd_op->output_df());

    if (epoch_type == graphlib::NodeEpochType::Backward)
    {
        node->set_backward();
        add_fwd_to_bwd_map(current_fwd_op, node, operand_index);
    }
    else if (epoch_type == graphlib::NodeEpochType::Optimizer)
    {
        node->set_optimizer();
        add_fwd_to_optimizer_edge(current_fwd_op, node, operand_index);
    }
    tag_disable_consteval(disable_consteval, node);

    return NodeContext(node);
}

// From the node in the forward graph, create a recompute node for the backward graph.
// This will also connect the newly created recompute node to its operands (based on the operands of the fwd node).
// NOTE: it is expected that the create_recompute_op() is called in topological order.
static Node *create_recompute_op(
    graphlib::Graph *graph, Node *fwd_node, std::unordered_map<Node *, Node *> &fwd_to_recompute)
{
    TT_ASSERT(fwd_node->is_forward(), "Expected a forward node");
    TT_ASSERT(fwd_node->node_type() == graphlib::NodeType::kPyOp, "Expected a PyOp node");

    auto cloned_node = fwd_node->clone(fwd_node->name() + "_recompute");
    cloned_node->set_epoch_type(graphlib::NodeEpochType::Backward);

    auto recompute_node = graph->add_node(std::move(cloned_node), graph->get_subgraph_id_for_node(fwd_node->id()));

    std::unordered_map<graphlib::NodeId, graphlib::NodeId> producer_remap;

    fwd_to_recompute[fwd_node] = recompute_node;

    // Hook up Recompute Node to its operands
    for (const Edge &edge : graph->operand_data_edges(fwd_node))
    {
        Node *fwd_operand = graph->node_by_id(edge.producer_node_id);
        bool same_epoch = (fwd_operand->is_forward() && fwd_operand->node_type() != graphlib::NodeType::kInput);

        TT_ASSERT(
            !same_epoch || fwd_to_recompute.find(fwd_operand) != fwd_to_recompute.end(),
            "Expected to find the analog recompute node for the operand");
        graphlib::NodeId producer_id = same_epoch ? fwd_to_recompute.at(fwd_operand)->id() : fwd_operand->id();

        if (producer_remap.find(producer_id) != producer_remap.end())
        {
            producer_id = producer_remap.at(producer_id);
        }

        Edge recompute_edge = Edge(
            producer_id,
            edge.producer_output_port_id,
            recompute_node->id(),
            edge.consumer_input_port_id,
            edge.edge_type);

        graph->add_edge(recompute_edge);

        auto fwd_edge_attr = graph->get_edge_attributes(edge);
        auto recompute_edge_attr = graph->get_edge_attributes(recompute_edge);
        recompute_edge_attr->copy_from(*fwd_edge_attr);
    }

    return recompute_node;
}

static void add_edges_for_recompute_op(
    graphlib::Graph *graph, const graphlib::Edge &fwd_to_bwd_edge, std::unordered_map<Node *, Node *> &fwd_to_recompute)
{
    Node *fwd_node = graph->node_by_id(fwd_to_bwd_edge.producer_node_id);

    if (fwd_to_recompute.find(fwd_node) == fwd_to_recompute.end())
    {
        return;
    }

    Node *recompute_node = fwd_to_recompute.at(fwd_node);

    // Replace the original FWD->BWD edge with RECOMPUTE->BWD instead.
    graph->add_edge(
        Edge(
            recompute_node->id(),
            fwd_to_bwd_edge.producer_output_port_id,
            fwd_to_bwd_edge.consumer_node_id,
            fwd_to_bwd_edge.consumer_input_port_id,
            fwd_to_bwd_edge.edge_type),
        graph->get_edge_attributes(fwd_to_bwd_edge));
    graph->remove_edge(fwd_to_bwd_edge);
}

void autograd_engine::insert_recompute_ops()
{
    TT_ASSERT(config.recompute, "Recompute is not enabled");

    std::unordered_map<Node *, Node *> forward_to_recompute;
    std::vector<Node *> topo_order = graphlib::topological_sort(*graph);

    // We loop through only the backward nodes and look for operand nodes that are marked
    // as FWD. Each instance is an opportunity to perform a potential recompute on the original
    // FWD op.
    //
    // TODO(#395): extend this to support configurable op-specific recompute; rather than an all-or-nothing sort-of
    // deal.
    std::deque<std::string> recompute_node_names;

    for (Node *node : topo_order)
    {
        if (node->is_forward() && node->node_type() == graphlib::NodeType::kPyOp)
        {
            Node *recompute_node = create_recompute_op(graph, node, forward_to_recompute);
            recompute_node_names.push_back(recompute_node->name());
        }
    }

    for (Node *bwd_node : topo_order)
    {
        if (!bwd_node->is_backward())
        {
            continue;
        }

        std::vector<Edge> bwd_operand_data_edges = graph->operand_data_edges(bwd_node);
        for (const Edge &operand_edge : bwd_operand_data_edges)
        {
            add_edges_for_recompute_op(graph, operand_edge, forward_to_recompute);
        }
    }

    std::unordered_map<Node *, std::string> forward_to_recompute_name;
    for (const auto [fwd_node, recompute_node] : forward_to_recompute)
    {
        forward_to_recompute_name[fwd_node] = recompute_node->name();
    }

    // Remove any recompute nodes that do not have any users; also removes chains of recompute nodes,
    // e.g. add_recompute -> mul_recompute -> nothing (both add and mul recompute nodes will be removed)
    while (not recompute_node_names.empty())
    {
        std::string recompute_node_name = recompute_node_names.front();
        recompute_node_names.pop_front();

        if (graph->has_node_with_name(recompute_node_name))
        {
            Node *recompute_node = graph->get_node_by_name(recompute_node_name);
            if (graph->num_users(recompute_node->id()) == 0)
            {
                for (Node *operand : graph->data_operands(recompute_node))
                {
                    recompute_node_names.push_back(operand->name());
                }
                graph->remove_node(recompute_node);
            }
        }
    }

    for (const auto &[fwd_node, recompute_node_name] : forward_to_recompute_name)
    {
        if (graph->has_node_with_name(recompute_node_name))
        {
            graph->add_edge(
                fwd_node, graph->get_node_by_name(recompute_node_name), graphlib::EdgeType::kAutogradFwdToRecompute);
        }
    }
    return;
}

}  // namespace autograd

}  // namespace tt
