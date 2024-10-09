// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "forge_graph_module.hpp"
#include "passes/split_graph.hpp"
#include "test/common.hpp"

namespace tt::test
{
struct SplitGraphTest
    : public ForgeGraphTest,
      public testing::WithParamInterface<bool>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        int batch_size = 32;
        int input_size = 784;
        int output_size = 10;
        int hidden = 256;

        auto act = create_activation(shape(1, 1, batch_size, input_size));

        constexpr bool requires_grad = true;

        auto weight_l1 = create_parameter(shape(1, 1, hidden, input_size), requires_grad);
        auto bias_l1 = create_parameter(shape(1, 1, 1, hidden), requires_grad);

        auto weight_l2 = create_parameter(shape(1, 1, output_size, hidden), requires_grad);
        auto bias_l2 = create_parameter(shape(1, 1, 1, output_size), requires_grad);

        auto transposed_weight_l1 = create_op("transpose", {weight_l1}, {{"dim0", -2}, {"dim1", -1}});
        auto l1 = create_op("matmul", {act, transposed_weight_l1});
        matmul_op_name = l1->name();

        auto add = create_op("add", {l1, bias_l1});

        auto transposed_weight_l2 = create_op("transpose", {weight_l2}, {{"dim0", -2}, {"dim1", -1}});
        auto l2 = create_op("matmul", {add, transposed_weight_l2});
        matmul_op_name = l2->name();

        auto add2 = create_op("add", {l2, bias_l2});

        auto softmax = create_op("softmax", {add2}, {{"dimension", 1}, {"keep_dim", true}}, 1, true);

        return {softmax};
    }

    std::string matmul_op_name;
};

using Graph = graphlib::Graph;

// Tests that the invariants from the original graph (specifically forward subgraph)
// are perserved in the extracted forward graph.
TEST_P(SplitGraphTest, test_forward)
{
    Graph* graph = get_graph();
    graph->set_training(true);
    Graph* original_graph = graph->clone();

    // TODO(#366): Recompute parameter in autograd config is not used.
    // So the variant of the test which uses recompute = true is still not actually testing training with recompute.
    bool recompute = GetParam();
    autograd::autograd_config config{.recompute = recompute, .optimizer = py::none()};

    auto autograd_engine = tt::autograd::autograd_engine(graph, config);
    graph = autograd_engine.run();

    EXPECT_TRUE(graph->contains_bwd_nodes());

    auto graph_module = passes::split_graph(graph);

    auto fwd_graph = graph_module.get_graph(tt::GraphType::Forward);

    // Verify that all nodes from the initial graph are present in the
    // extracted forward graph.
    for (auto node : original_graph->nodes())
    {
        EXPECT_TRUE(fwd_graph->has_node_with_name(node->name()));
    }

    // Verify ordered module inputs.
    // We need to perserve the original order of module inputs, throughout the transformations and passes.
    // The order of module inputs is determined by the user in the python code (torch, tf, etc.) and
    // is set by us when creating initial graph.
    for (auto input : original_graph->ordered_module_inputs())
    {
        EXPECT_TRUE(fwd_graph->has_node_with_name(input->name()));
    }
    EXPECT_EQ(original_graph->get_ordered_input_names(), fwd_graph->get_ordered_input_names());
}

// Tests that the invariants from the original graph (specifically backward subgraph)
// are perserved in the extracted backward graph.
TEST_P(SplitGraphTest, test_backward)
{
    Graph* graph = get_graph();
    graph->set_training(true);

    // TODO(#366): Recompute parameter in autograd config is not used.
    // So the variant of the test which uses recompute = true is still not actually testing training with recompute.
    bool recompute = GetParam();
    autograd::autograd_config config{.recompute = recompute, .optimizer = py::none()};

    auto autograd_engine = tt::autograd::autograd_engine(graph, config);
    graph = autograd_engine.run();

    EXPECT_TRUE(graph->contains_bwd_nodes());

    auto graph_module = passes::split_graph(graph);

    auto fwd_graph = graph_module.get_graph(tt::GraphType::Forward);
    auto bwd_graph = graph_module.get_graph(tt::GraphType::Backward);
    EXPECT_TRUE(bwd_graph != nullptr);

    // Verify that all intermediate outputs from the forward graph are inputs
    // to the backward graph.
    for (auto intermediate_output : fwd_graph->get_ordered_intermediate_names())
    {
        EXPECT_TRUE(fwd_graph->get_node_by_name(intermediate_output)->node_type() == graphlib::NodeType::kOutput);
        EXPECT_TRUE(bwd_graph->has_node_with_name(intermediate_output));
        EXPECT_TRUE(bwd_graph->get_node_by_name(intermediate_output)->node_type() == graphlib::NodeType::kInput);
    }

    // Verify that all nodes from the backward graph have the same number
    // of operands as in the original graph (containing both forward and backward subgraphs).
    auto is_bwd_op_node = [](graphlib::Node* node) { return node->is_backward() && node->node_type() == graphlib::NodeType::kPyOp; };
    for (auto node : graph->nodes(is_bwd_op_node))
    {
        EXPECT_TRUE(bwd_graph->has_node_with_name(node->name()));
        auto bwd_node = bwd_graph->get_node_by_name(node->name());
        EXPECT_EQ(graph->data_operands(node).size(), bwd_graph->data_operands(bwd_node).size()) << "Node " << node->name() << " has different number of operands in the backward graph";
    }
}

INSTANTIATE_TEST_SUITE_P(
    SplitGraphTest,
    SplitGraphTest,
    ::testing::Values(true, false)
);

}  // namespace tt::test
