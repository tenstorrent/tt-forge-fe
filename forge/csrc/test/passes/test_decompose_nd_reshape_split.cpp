// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/edge.hpp"
#include "graph_lib/node_types.hpp"
#include "gtest/gtest.h"
#include "passes/decompose_nd_reshape_split.hpp"
#include "reportify/reportify.hpp"
#include "test/graph_api.hpp"

using namespace tt;

struct DecomposeNdReshapeSplitTest : testing::Test
{
    graphlib::Graph *graph;

    DecomposeNdReshapeSplitTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "DecomposeNdReshapeSplitTest");
    }

    ~DecomposeNdReshapeSplitTest() { delete graph; }
};

TEST_F(DecomposeNdReshapeSplitTest, basic_dimension_split_optimization)
{
    // Test case: Input (2, 12) -> Reshape (2, 2, 6) -> 2x Index -> 2x Squeeze -> (2, 6)
    // Expected: Input (2, 12) -> 2x Index (used like slice) -> 2x Reshape(nop) -> (2, 6)

    // Create input with shape (2, 12)
    auto input_node = create_input(*graph, "input", graphlib::Shape::create({2, 12}));

    // Create reshape: (2, 12) -> (2, 2, 6)
    auto reshape_node = add_node<graphlib::PyOpNode>(*graph, "reshape", "reshape", {2, 2, 6}, {input_node});
    reshape_node->set_shape(graphlib::Shape::create({2, 2, 6}));

    // Create index operations
    auto index1_node = add_node<graphlib::PyOpNode>(*graph, "index1", "index", {1, 0, 1, 1}, {reshape_node});
    index1_node->set_shape(graphlib::Shape::create({2, 1, 6}));

    auto index2_node = add_node<graphlib::PyOpNode>(*graph, "index2", "index", {1, 1, 2, 1}, {reshape_node});
    index2_node->set_shape(graphlib::Shape::create({2, 1, 6}));

    // Create squeeze operations
    auto squeeze1_node = add_node<graphlib::PyOpNode>(*graph, "squeeze1", "reshape", {2, 6}, {index1_node});
    squeeze1_node->set_shape(graphlib::Shape::create({2, 6}));

    auto squeeze2_node = add_node<graphlib::PyOpNode>(*graph, "squeeze2", "reshape", {2, 6}, {index2_node});
    squeeze2_node->set_shape(graphlib::Shape::create({2, 6}));

    // Create outputs
    create_output(*graph, "out1", squeeze1_node);
    create_output(*graph, "out2", squeeze2_node);

    // Count operations before pass
    int index_count_before = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == graphlib::kPyOp)
        {
            auto op_node = node->as<graphlib::PyOpNode>();
            if (op_node->op_name() == "index")
                index_count_before++;
        }
    }
    EXPECT_EQ(index_count_before, 2) << "Should start with 2 index operations";

    // Run the pass
    passes::decompose_nd_reshape_split(graph);

    // Verify the transformation

    // 1. The main reshape should be bypassed/removed
    auto reshape_it = std::find_if(
        graph->nodes().begin(), graph->nodes().end(), [](graphlib::Node *n) { return n->name() == "reshape"; });
    EXPECT_EQ(reshape_it, graph->nodes().end()) << "Main reshape should be bypassed/removed";

    // 2. Should still have 2 index operations
    int index_count_after = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == graphlib::kPyOp)
        {
            auto op_node = node->as<graphlib::PyOpNode>();
            if (op_node->op_name() == "index")
                index_count_after++;
        }
    }
    EXPECT_EQ(index_count_after, 2) << "Should have 2 index operations after pass";

    // 3. Verify index operations connect directly to input and have correct attributes
    for (auto node : graph->nodes())
    {
        if (node->node_type() == graphlib::kPyOp)
        {
            auto op_node = node->as<graphlib::PyOpNode>();
            if (op_node->op_name() == "index")
            {
                // Check connection to input
                auto operands = graph->operands(node);
                EXPECT_EQ(operands.size(), 1) << "Index should have one operand";
                EXPECT_EQ(operands[0]->name(), "input") << "Index should connect directly to input";

                // Check attributes
                auto attrs = op_node->op_legacy_attrs();
                EXPECT_EQ(attrs.size(), 4) << "Index should have 4 attributes";

                int dim = std::get<int>(attrs[0]);
                int start = std::get<int>(attrs[1]);
                int stop = std::get<int>(attrs[2]);
                int stride = std::get<int>(attrs[3]);

                EXPECT_EQ(dim, 1) << "Index should operate on dimension 1";
                EXPECT_EQ(stop - start, 6) << "Index slice size should be 6";
                EXPECT_EQ(stride, 1) << "Index stride should be 1";
                EXPECT_TRUE(start == 0 || start == 6) << "Index start should be 0 or 6";
                EXPECT_TRUE(stop == 6 || stop == 12) << "Index stop should be 6 or 12";
            }
        }
    }
}

TEST_F(DecomposeNdReshapeSplitTest, invalid_cases_should_be_skipped)
{
    // Test cases that should NOT be optimized by the pass

    // Rank change > 1 (should be skipped)
    auto input1 = create_input(*graph, "input1", graphlib::Shape::create({2, 12}));
    auto reshape1 = add_node<graphlib::PyOpNode>(*graph, "reshape1", "reshape", {2, 2, 2, 3}, {input1});
    reshape1->set_shape(graphlib::Shape::create({2, 2, 2, 3}));
    create_output(*graph, "out1", reshape1);

    int node_count_before = graph->nodes().size();

    // Run the pass
    passes::decompose_nd_reshape_split(graph);

    int node_count_after = graph->nodes().size();

    // Should be no change since these cases should be skipped
    EXPECT_EQ(node_count_before, node_count_after) << "Invalid cases should not be modified";
}
