// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <type_traits>
#include <unordered_map>

#include "gtest/gtest.h"
#include "test/graph_api.hpp"

namespace tt
{
struct DeviceConfig;
}

namespace tt::test
{
template <graphlib::IRLevel ir_level>
class GraphTest : public ::testing::Test
{
   public:
    using OpType =
        std::conditional_t<ir_level == graphlib::IRLevel::IR_FORGE, graphlib::ForgeOpNode, graphlib::PyOpNode>;

    virtual std::vector<OpType*> create_graph() = 0;

    void SetUp() override
    {
        graph = std::make_unique<graphlib::Graph>(ir_level, std::string("GraphTest.") + get_current_test_name());
        for (OpType* output : create_graph())
        {
            create_output(output);
        }
        graph->dump("pre");
    }

    void TearDown() override { graph->dump("post"); }

    graphlib::Graph* get_graph()
    {
        TT_ASSERT(output_name_id > 0, "Must have created an output node before graph is legal");
        return graph.get();
    }

    static std::string get_current_test_name()
    {
        std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        for (std::string bad_char : std::vector{"/"})
            for (std::string::size_type n = name.find(bad_char, 0); n != std::string::npos; n = name.find(bad_char, n))
                name.replace(n, bad_char.size(), "_");
        return name;
    }

    template <typename... Dims>
    static graphlib::Shape shape(Dims... dims)
    {
        if constexpr (ir_level == graphlib::IRLevel::IR_FORGE)
        {
            return graphlib::Shape::create_forge({static_cast<std::uint32_t>(dims)...});
        }
        else
        {
            return graphlib::Shape::create({static_cast<std::uint32_t>(dims)...});
        }
    }

    graphlib::InputNode* create_input(
        const std::string& name, tt::graphlib::Shape const& shape, tt::graphlib::InputNodeType input_type)
    {
        return tt::create_input(*graph, name, shape, input_type);
    }

    graphlib::InputNode* create_input(graphlib::Shape const& shape, tt::graphlib::InputNodeType input_type)
    {
        auto name = graphlib::to_string(input_type) + std::to_string(input_name_id++);
        return tt::create_input(*graph, name, shape, input_type);
    }

    graphlib::InputNode* create_activation(graphlib::Shape const& shape)
    {
        auto node = create_input(shape, tt::graphlib::InputNodeType::Activation);
        graph->register_module_inputs({node->id()}, true /* append */);
        return node;
    }

    template <typename... Dims>
    graphlib::InputNode* create_activation(Dims... dims)
    {
        return create_activation(shape(dims...));
    }

    graphlib::InputNode* create_parameter(graphlib::Shape const& shape, bool requires_grad = false)
    {
        auto node = create_input(shape, tt::graphlib::InputNodeType::Parameter);
        node->set_requires_grad(requires_grad);
        return node;
    }

    template <typename... Dims>
    graphlib::InputNode* create_parameter(Dims... dims)
    {
        return create_activation(shape(dims...));
    }

    graphlib::InputNode* create_constant(graphlib::Shape const& shape)
    {
        return create_input(shape, tt::graphlib::InputNodeType::Constant);
    }

    template <typename... Dims>
    graphlib::InputNode* create_constant(Dims... dims)
    {
        return create_constant(shape(dims...));
    }

    OpType* create_op(
        std::string const& name, graphlib::OpType const& op_type, std::vector<graphlib::Node*> const& operands)
    {
        return tt::add_node<OpType>(
            *graph, name, op_type.op, op_type.attr, operands, {}, op_type.forge_attrs, op_type.named_attrs);
    }

    OpType* create_op(graphlib::OpType const& op_type, std::vector<graphlib::Node*> const& operands)
    {
        auto name = op_type.op + std::to_string(op_name_id[op_type.op]++);
        return create_op(name, op_type, operands);
    }

    OpType* create_op(
        std::string const& type,
        std::vector<graphlib::Node*> const& operands,
        std::vector<graphlib::OpType::Attr> op_attrs)
    {
        return create_op(graphlib::OpType(type, op_attrs), operands);
    }

    OpType* create_op(
        std::string const& name,
        std::string const& type,
        std::vector<graphlib::Node*> const& operands,
        std::vector<graphlib::OpType::Attr> op_attrs)
    {
        return create_op(name, graphlib::OpType(type, op_attrs), operands);
    }

    template <typename... Attrs>
    OpType* create_op(
        std::string const& type,
        std::vector<graphlib::Node*> const& operands,
        const tt::ForgeOpAttrs& forge_attrs,
        Attrs... attrs)
    {
        return create_op(graphlib::OpType(type, {attrs...}, forge_attrs), operands);
    }

    template <typename... Attrs>
    OpType* create_op(
        std::string const& name,
        std::string const& type,
        std::vector<graphlib::Node*> const& operands,
        const tt::ForgeOpAttrs& forge_attrs,
        Attrs... attrs)
    {
        return create_op(name, graphlib::OpType(type, {attrs...}, forge_attrs), operands);
    }

    OpType* create_op(
        std::string const& type,
        std::vector<graphlib::Node*> const& operands,
        const graphlib::OpType::Attrs& named_attrs)
    {
        return create_op(graphlib::OpType(type, {}, {}, named_attrs), operands);
    }

    OpType* create_op(
        std::string const& name,
        std::string const& type,
        std::vector<graphlib::Node*> const& operands,
        const graphlib::OpType::Attrs& named_attrs)
    {
        return create_op(name, graphlib::OpType(type, {}, {}, named_attrs), operands);
    }

    template <typename... Attrs>
    OpType* create_op(std::string const& type, std::vector<graphlib::Node*> const& operands, Attrs... attrs)
    {
        return create_op(graphlib::OpType(type, {attrs...}), operands);
    }

    template <typename... Attrs>
    OpType* create_op(
        std::string const& name, std::string const& type, std::vector<graphlib::Node*> const& operands, Attrs... attrs)
    {
        return create_op(name, graphlib::OpType(type, {attrs...}), operands);
    }

    graphlib::QueueNode* create_queue(graphlib::Node* operand, bool cross_epoch = false, bool cross_chip = false)
    {
        auto name = "queue" + std::to_string(op_name_id["queue"]++);
        graphlib::QueueNode* queue = graph->add_node(
            graphlib::create_node<graphlib::EpochToEpochQueueNode>(name, cross_epoch, cross_chip),
            graph->get_subgraph_id_for_node(operand->id()));
        tt::add_operand_edges(*graph, queue, {operand});
        return queue;
    }

    graphlib::QueueNode* create_buffering_queue(graphlib::Node* operand, int num_entries)
    {
        auto name = "buff_queue" + std::to_string(op_name_id["buff_queue"]++);
        graphlib::QueueNode* queue = graph->add_node(
            graphlib::create_node<graphlib::BufferingQueueNode>(name, num_entries),
            graph->get_subgraph_id_for_node(operand->id()));
        tt::add_operand_edges(*graph, queue, {operand});
        return queue;
    }

    template <typename... Attrs>
    graphlib::QueueNode* create_queue(graphlib::Node* operand, Attrs... attrs)
    {
        return create_queue(operand, attrs...);
    }

    void append_tm(graphlib::OpType const& op_type, std::shared_ptr<graphlib::EdgeAttributes> attr)
    {
        TT_ASSERT(ir_level == graphlib::IRLevel::IR_FORGE);
        attr->append_tm(op_type);
    }

    void append_tm(graphlib::OpType const& op_type, graphlib::Edge edge)
    {
        return append_tm(op_type, graph->get_edge_attributes(edge));
    }

    void append_tm(graphlib::OpType const& op_type, graphlib::Node* node, int operand_idx)
    {
        auto edge = graph->operand_data_edges(node).at(operand_idx);
        return append_tm(op_type, graph->get_edge_attributes(edge));
    }

    template <typename... Attrs>
    void append_tm(std::string const& type, graphlib::Edge edge, Attrs... attrs)
    {
        return append_tm(graphlib::OpType(type, {attrs...}), edge);
    }

    template <typename... Attrs>
    void append_tm(std::string const& type, graphlib::Node* node, int operand_idx, Attrs... attrs)
    {
        return append_tm(graphlib::OpType(type, {attrs...}), node, operand_idx);
    }

    void append_tm(
        std::string const& type, graphlib::Node* node, int operand_idx, graphlib::OpType::Attrs const& named_attrs)
    {
        return append_tm(graphlib::OpType(type, {}, {}, named_attrs), node, operand_idx);
    }

    std::vector<graphlib::OpType>& get_tms(std::shared_ptr<graphlib::EdgeAttributes> attr)
    {
        TT_ASSERT(ir_level == graphlib::IRLevel::IR_FORGE);
        return attr->get_tms();
    }

    std::vector<graphlib::OpType>& get_tms(graphlib::Edge edge) { return get_tms(graph->get_edge_attributes(edge)); }

    std::vector<graphlib::OpType>& get_tms(graphlib::Node* node, int operand_idx)
    {
        return get_tms(graph->get_edge_attributes(graph->operand_data_edges(node).at(operand_idx)));
    }

    graphlib::OutputNode* create_output(graphlib::Node* operand)
    {
        auto name = "output" + std::to_string(output_name_id++);
        return tt::create_output(*graph, name, operand);
    }

   private:
    std::unique_ptr<graphlib::Graph> graph;
    int input_name_id = 0;
    int output_name_id = 0;
    std::unordered_map<std::string, int> op_name_id;
};

using ForgeGraphTest = GraphTest<graphlib::IRLevel::IR_TT_FORGE>;

}  // namespace tt::test
