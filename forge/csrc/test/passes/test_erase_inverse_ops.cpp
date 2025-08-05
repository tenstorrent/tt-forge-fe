// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/edge.hpp"
#include "graph_lib/node_types.hpp"
#include "gtest/gtest.h"
#include "lower_to_forge/common.hpp"
#include "ops/op.hpp"
#include "passes/commute_utils.hpp"
#include "passes/erase_inverse_ops.hpp"
#include "passes/insert_inverse_on_io.hpp"
#include "passes/replace_incommutable_patterns.hpp"
#include "reportify/reportify.hpp"
#include "test/graph_api.hpp"

using namespace tt;

struct EraseInverseOps : testing::Test
{
    graphlib::Graph *graph;

    EraseInverseOps()
    {
        // Two transposes feeding into eltwise which has a transpose after it
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE);

        graphlib::Shape shape = graphlib::Shape::create({1, 1, 512, 160});
        graphlib::Shape shapeT = graphlib::Shape::create({1, 1, 160, 512});

        auto in0_a = create_input(*graph, "in0_a", graphlib::Shape::create({1, 1, shape[2], 256}));
        auto in0_b = create_input(*graph, "in0_b", graphlib::Shape::create({1, 1, 256, shape[3]}));
        auto matmul0 = add_node<graphlib::PyOpNode>(*graph, "matmul0", "matmul", {}, {in0_a, in0_b});
        auto transpose0 = add_node<graphlib::PyOpNode>(
            *graph, "transpose0", graphlib::OpType("transpose", {}, {{"dim0", 0}, {"dim1", 1}}), {matmul0});

        auto in1_a = create_input(*graph, "in1_a", graphlib::Shape::create({1, 1, shape[2], 128}));
        auto in1_b = create_input(*graph, "in1_b", graphlib::Shape::create({1, 1, 128, shape[3]}));
        auto matmul1 = add_node<graphlib::PyOpNode>(*graph, "matmul1", "matmul", {}, {in1_a, in1_b});
        auto transpose1 = add_node<graphlib::PyOpNode>(
            *graph, "transpose1", graphlib::OpType("transpose", {}, {{"dim0", 0}, {"dim1", 1}}), {matmul1});

        auto add = add_node<graphlib::PyOpNode>(*graph, "add", "add", {}, {transpose0, transpose1});
        auto post_transpose = add_node<graphlib::PyOpNode>(
            *graph, "post_transpose", graphlib::OpType("transpose", {}, {{"dim0", 0}, {"dim1", 1}}), {add});

        create_output(*graph, "out0", post_transpose);
    }
};

TEST_F(EraseInverseOps, erase_transpose)
{
    passes::erase_inverse_ops(graph);

    // Transposes should be gone
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->new_op_type(), ops::OpType::Transpose);
        }
    }
    EXPECT_EQ(graph->nodes().size(), 8);
}

// Two transposes feeding into eltwise which has a fork to another tranpose but also a fork-join buffer
TEST_F(EraseInverseOps, erase_transpose_fork)
{
    // fork after add into a transpose and a buffer
    graphlib::Node *add = graph->get_node_by_name("add");
    auto buffer = add_node<graphlib::PyOpNode>(*graph, "buffer", "nop", {}, {add});

    create_output(*graph, "out1", buffer);

    bool attempt_update = true;
    while (attempt_update)
    {
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update)
        {
            attempt_update = passes::insert_inverse_on_outputs(graph);
            if (not attempt_update)
            {
                attempt_update = passes::insert_inverse_on_inputs(graph);
            }
        }
    }

    // Because intermediate value is read out, we have to transpose the output on the out0
    int transpose_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->new_op_type() == ops::OpType::Transpose)
                transpose_count++;
        }
    }
    EXPECT_EQ(transpose_count, 1);

    EXPECT_EQ(graph->nodes().size(), 11);
}

// Two transposes feeding into eltwise which has a fork to another tranpose but also a fork-join buffer. Eventually they
// are joined.
TEST_F(EraseInverseOps, erase_inverse_ops_transpose_fork_join)
{
    // fork after add into a transpose and a buffer
    graphlib::Node *add = graph->get_node_by_name("add");
    auto buffer1 = add_node<graphlib::PyOpNode>(*graph, "buffer1", "nop", {}, {add});
    auto buffer2 = add_node<graphlib::PyOpNode>(*graph, "buffer2", "nop", {}, {buffer1});

    auto post_transpose = graph->get_node_by_name("post_transpose");
    auto unary = add_node<graphlib::PyOpNode>(*graph, "unary", "exp", {}, {post_transpose});
    auto unary_transpose = add_node<graphlib::PyOpNode>(
        *graph, "unary_transpose", graphlib::OpType("transpose", {}, {{"dim0", 0}, {"dim1", 1}}), {unary});
    auto join = add_node<graphlib::PyOpNode>(*graph, "join", "add", {}, {unary_transpose, buffer2});

    graph->remove_node(graph->get_node_by_name("out0"));
    create_output(*graph, "out0", join);

    bool attempt_update = true;
    while (attempt_update)
    {
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update)
        {
            attempt_update = passes::insert_inverse_on_outputs(graph);
            if (not attempt_update)
            {
                attempt_update = passes::insert_inverse_on_inputs(graph);
            }
        }
    }

    // Because intermediate value is read out, we have to transpose the output on the out0
    int transpose_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->new_op_type() == ops::OpType::Transpose)
                transpose_count++;
        }
    }
    EXPECT_EQ(transpose_count, 1);

    EXPECT_EQ(graph->nodes().size(), 13);
}

TEST_F(EraseInverseOps, erase_inverse_ops_dual_reduce)
{
    // fork after add into a transpose and a buffer
    // graphlib::Node *add = graph->get_node_by_name("add");
    // auto buffer1 = add_node<graphlib::PyOpNode>(*graph, "buffer1", "nop", {}, {add});
    // auto buffer2 = add_node<graphlib::PyOpNode>(*graph, "buffer2", "nop", {}, {buffer1});

    auto post_transpose = graph->get_node_by_name("post_transpose");
    auto smx_1 = add_node<graphlib::PyOpNode>(
        *graph, "smx_1", graphlib::OpType("softmax", {}, {{"dim", -1}, {"stable", false}}), {post_transpose});
    auto reshape_1 = add_node<graphlib::PyOpNode>(
        *graph, "reshape_1", graphlib::OpType("reshape", {}, {{"shape", std::vector{1, 512, 10, 16}}}), {smx_1});

    auto reduce_1 = add_node<graphlib::PyOpNode>(
        *graph,
        "reduce_1",
        graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{-2}}, {"keep_dim", true}}),
        {reshape_1});
    auto reduce_2 = add_node<graphlib::PyOpNode>(
        *graph,
        "reduce_2",
        graphlib::OpType("reduce_sum", {}, {{"dim_arg", std::vector<int>{-1}}, {"keep_dim", true}}),
        {reduce_1});
    auto reshape_2 = add_node<graphlib::PyOpNode>(
        *graph, "reshape_2", graphlib::OpType("reshape", {}, {{"shape", std::vector{1, 1, 512, 1}}}), {reduce_2});

    auto smx_2 = add_node<graphlib::PyOpNode>(
        *graph, "smx_2", graphlib::OpType("softmax", {}, {{"dim", -1}, {"stable", false}}), {reshape_2});
    graph->remove_node(graph->get_node_by_name("out0"));
    create_output(*graph, "out0", smx_2);

    bool attempt_update = true;
    while (attempt_update)
    {
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update)
        {
            attempt_update = passes::insert_inverse_on_outputs(graph);
            if (not attempt_update)
            {
                attempt_update = passes::insert_inverse_on_inputs(graph);
            }
        }
    }

    // Because intermediate value is read out, we have to transpose the output on the out0
    int transpose_count = 0;
    int reduce_count = 0;
    int reshape_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->new_op_type() == ops::OpType::Transpose)
                transpose_count++;
            else if (node->as<graphlib::PyOpNode>()->new_op_type() == ops::OpType::ReduceSum)
                reduce_count++;
            else if (node->as<graphlib::PyOpNode>()->new_op_type() == ops::OpType::Reshape)
                reshape_count++;
        }
    }
    EXPECT_EQ(transpose_count, 0);
    EXPECT_EQ(reduce_count, 1);
    EXPECT_EQ(reshape_count, 0);

    // EXPECT_EQ(graph->nodes().size(), 13);
}

TEST_F(EraseInverseOps, replace_x_y_change_concat_pattern)
{
    auto post_transpose = graph->get_node_by_name("post_transpose");
    auto reshape_0 = add_node<graphlib::PyOpNode>(
        *graph, "reshape_0", graphlib::OpType("reshape", {}, {{"shape", std::vector{256, 320}}}), {post_transpose});
    auto reshape_1 = add_node<graphlib::PyOpNode>(
        *graph, "reshape_1", graphlib::OpType("reshape", {}, {{"shape", std::vector{256, 320}}}), {post_transpose});
    auto reshape_2 = add_node<graphlib::PyOpNode>(
        *graph, "reshape_2", graphlib::OpType("reshape", {}, {{"shape", std::vector{256, 320}}}), {post_transpose});
    auto concat = add_node<graphlib::PyOpNode>(
        *graph, "concat", graphlib::OpType("concatenate", {}, {{"dim", -2}}), {reshape_0, reshape_1, reshape_2});

    graph->remove_node(graph->get_node_by_name("out0"));
    create_output(*graph, "out0", concat);

    passes::replace_incommutable_patterns(
        graph);                        // Should insert inverses under each reshape and a single clone after the concat
    passes::erase_inverse_ops(graph);  // Should erase all inverses, leaving just the clone after he concat

    int reshape_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->new_op_type() == ops::OpType::Reshape)
                reshape_count++;
        }
    }

    EXPECT_EQ(reshape_count, 1);
}

struct CommuteBroadcastThroughTranspose : testing::Test
{
    graphlib::Graph *graph;

    CommuteBroadcastThroughTranspose()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "CommuteBroadcastThroughTranspose");

        graphlib::Shape shape = graphlib::Shape::create({64, 1, 1});
        graphlib::Shape shapeT = graphlib::Shape::create({1, 112, 64, 112});

        tt::graphlib::InputNode *in0_a = create_input(*graph, "in0_a", shape, graphlib::InputNodeType::Constant);
        graphlib::PyOpNode *transpose = add_node<graphlib::PyOpNode>(
            *graph, "transpose", graphlib::OpType("transpose", {}, {{"dim0", -3}, {"dim1", -2}}), {in0_a});

        // There is only one edge between in0_a and transpose nodes.
        graphlib::Edge edge_with_bcst = graph->get_edges(in0_a, transpose)[0];
        // Add broadcast to the edge
        graph->get_edge_attributes(edge_with_bcst)->set_broadcast_dim(-2, 112);
        graph->get_edge_attributes(edge_with_bcst)->set_broadcast_dim(-1, 112);

        tt::graphlib::InputNode *in1_b = create_input(*graph, "in1_b", shapeT);
        graphlib::PyOpNode *multiply =
            add_node<graphlib::PyOpNode>(*graph, "multiply", "multiply", {}, {transpose, in1_b});

        create_output(*graph, "out0", multiply);
    }
};

TEST_F(CommuteBroadcastThroughTranspose, commute_broadcast_through_transpose)
{
    Node *transpose = graph->get_node_by_name("transpose");
    Node *multiply = graph->get_node_by_name("multiply");

    // Commute broadcast through transpose
    graphlib::OpNode *op_node_transpose = dynamic_cast<graphlib::OpNode *>(transpose);
    EXPECT_TRUE(op_node_transpose != nullptr);
    bool commuted = passes::try_commute_bcast_through_clone(graph, op_node_transpose);
    EXPECT_TRUE(commuted);

    // Check if the broadcast dim is updated
    graphlib::Edge edge = graph->get_edges(transpose, multiply)[0];
    std::vector<graphlib::OpType> tms = graph->get_edge_attributes(edge)->get_tms();
    EXPECT_EQ(tms.size(), 2);
    EXPECT_EQ(tms[0].type(), ops::OpType::Broadcast);
    EXPECT_EQ(tms[1].type(), ops::OpType::Broadcast);
    // Broadcast along dimension -2 should become broadcast along -3
    // after commuting through transpose.
    EXPECT_EQ(tms[1].attr_as<int>("dim"), -3);
    EXPECT_EQ(tms[1].attr_as<int>("size"), 112);
}

struct UpdateReshapeNamedAttrsTest : testing::Test
{
    graphlib::Graph *graph;

    UpdateReshapeNamedAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateReshapeNamedAttrs");

        graphlib::Shape initial_shape = graphlib::Shape::create({1, 512, 160});
        tt::graphlib::InputNode *input_node = create_input(*graph, "input", initial_shape);

        auto reshape_node = add_node<graphlib::PyOpNode>(
            *graph,
            "reshape",
            graphlib::OpType("reshape", {}, {{"shape", std::vector{1, 1, 512 * 160}}}),
            {input_node});

        create_output(*graph, "out", reshape_node);
    }
};

TEST_F(UpdateReshapeNamedAttrsTest, update_named_attrs)
{
    graphlib::Node *reshape = graph->get_node_by_name("reshape");
    ASSERT_NE(reshape, nullptr) << "Reshape node not found.";

    graphlib::OpNode *op_node_reshape = dynamic_cast<graphlib::OpNode *>(reshape);
    ASSERT_NE(op_node_reshape, nullptr) << "Node is not of type OpNode.";

    graphlib::Shape new_shape = graphlib::Shape::create({1, 512, 160, 1});
    op_node_reshape->set_shape(new_shape);
    passes::update_reshape_attr(op_node_reshape, new_shape);

    auto updated_attrs = op_node_reshape->op_type().attrs();
    EXPECT_TRUE(updated_attrs.count("shape")) << "Shape attribute not found.";
    auto shape_vector = std::get<std::vector<int>>(updated_attrs["shape"]);

    std::vector<std::uint32_t> shape_vector_uint(shape_vector.begin(), shape_vector.end());
    graphlib::Shape updated_shape = graphlib::Shape::create(shape_vector_uint);

    // Compare updated_shape with new_shape
    EXPECT_EQ(updated_shape, new_shape) << "Shape attribute does not match expected value.";

    // Verify the node's shape
    EXPECT_EQ(op_node_reshape->shape(), new_shape);
}

struct UpdateSelectNamedAttrsTest : testing::Test
{
    graphlib::Graph *graph;

    UpdateSelectNamedAttrsTest()
    {
        int dim = 1;
        int begin = 0;
        int length = 5;
        int stride = 1;
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateSelectNamedAttrs");

        graphlib::Shape initial_shape = graphlib::Shape::create({1, 512, 160});
        graphlib::InputNode *input_node = create_input(*graph, "input", initial_shape);
        auto select_node = add_node<graphlib::PyOpNode>(
            *graph,
            "select",
            graphlib::OpType(
                "select",
                {dim, begin, length, stride},
                {{"dim", dim}, {"begin", begin}, {"length", length}, {"stride", stride}}),
            {input_node});
        select_node->set_op_attr("select_dim", dim);
        select_node->set_op_attr("begin", begin);
        select_node->set_op_attr("length", length);
        select_node->set_op_attr("stride", stride);

        create_output(*graph, "out", select_node);
    }
};

TEST_F(UpdateSelectNamedAttrsTest, update_named_attrs)
{
    graphlib::Node *select = graph->get_node_by_name("select");
    ASSERT_NE(select, nullptr) << "Select node not found.";

    graphlib::OpNode *op_node_select = dynamic_cast<graphlib::OpNode *>(select);
    ASSERT_NE(op_node_select, nullptr) << "Node is not of type OpNode.";

    int select_dim = 1;
    graphlib::Shape commute_shape = graphlib::Shape::create({1, 3, 512, 160});

    passes::update_select_attr(op_node_select, select_dim);

    auto updated_attrs = op_node_select->op_named_attrs();

    EXPECT_TRUE(updated_attrs.count("select_dim")) << "select_dim attribute not found.";
    EXPECT_EQ(std::get<int>(updated_attrs["select_dim"]), select_dim) << "select_dim does not match expected value.";
}

struct UpdateConcatNamedAttrsTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *reshape_0;
    graphlib::OpNode *reshape_1;
    graphlib::OpNode *reshape_2;

    UpdateConcatNamedAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateConcatNamedAttrs");

        graphlib::Shape shape_0 = graphlib::Shape::create({1, 512, 160});
        graphlib::Shape shape_1 = graphlib::Shape::create({1, 512, 160});
        graphlib::Shape shape_2 = graphlib::Shape::create({1, 512, 160});

        tt::graphlib::InputNode *input_node_0 = create_input(*graph, "input_0", shape_0);
        tt::graphlib::InputNode *input_node_1 = create_input(*graph, "input_1", shape_1);
        tt::graphlib::InputNode *input_node_2 = create_input(*graph, "input_2", shape_2);

        reshape_0 = add_node<graphlib::PyOpNode>(
            *graph,
            "reshape_0",
            graphlib::OpType("reshape", {}, {{"shape", std::vector{1, 1, 512 * 160}}}),
            {input_node_0});

        reshape_1 = add_node<graphlib::PyOpNode>(
            *graph,
            "reshape_1",
            graphlib::OpType("reshape", {}, {{"shape", std::vector{1, 1, 512 * 160}}}),
            {input_node_1});

        reshape_2 = add_node<graphlib::PyOpNode>(
            *graph,
            "reshape_2",
            graphlib::OpType("reshape", {}, {{"shape", std::vector{1, 1, 512 * 160}}}),
            {input_node_2});

        auto concat_node = add_node<graphlib::PyOpNode>(
            *graph, "concat", graphlib::OpType("concatenate", {}, {{"dim", -2}}), {reshape_0, reshape_1, reshape_2});

        create_output(*graph, "out", concat_node);
    }
};

TEST_F(UpdateConcatNamedAttrsTest, update_named_attrs)
{
    graphlib::Node *concat = graph->get_node_by_name("concat");
    ASSERT_NE(concat, nullptr) << "Concatenate node not found.";
    graphlib::OpNode *op_node_concat = dynamic_cast<graphlib::OpNode *>(concat);
    ASSERT_NE(op_node_concat, nullptr) << "Node is not of type OpNode.";
    int new_dim = 2;
    passes::update_concat_attr(op_node_concat, new_dim);
    auto updated_attrs = op_node_concat->op_named_attrs();
    EXPECT_TRUE(updated_attrs.count("dim")) << "Dim attribute not found.";
    auto dim_value = std::get<int>(updated_attrs["dim"]);
    EXPECT_EQ(dim_value, new_dim) << "Dim attribute does not match expected value.";
}

struct UpdateMatMulNamedAttrsTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *matmul;

    UpdateMatMulNamedAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateMatMulNamedAttrs");

        graphlib::Shape shape_0 = graphlib::Shape::create({1, 512, 512});
        graphlib::Shape shape_1 = graphlib::Shape::create({512, 512});

        auto input_node_0 = create_input(*graph, "input_0", shape_0);
        auto input_node_1 = create_input(*graph, "input_1", shape_1);

        matmul = add_node<graphlib::PyOpNode>(*graph, "matmul", "matmul", {}, {input_node_0, input_node_1});
        create_output(*graph, "output", matmul);
    }
};

struct UpdateConvAttrsTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *conv_node;

    UpdateConvAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateConvAttrsTest");

        graphlib::Shape shape_0 = graphlib::Shape::create({1, 256, 224, 224});
        tt::graphlib::InputNode *input_node_0 = create_input(*graph, "input_0", shape_0);

        graphlib::Shape weight_shape = graphlib::Shape::create({256, 256, 3, 3});
        tt::graphlib::InputNode *weight_node = create_input(*graph, "weight", weight_shape);

        conv_node = add_node<graphlib::PyOpNode>(
            *graph, "conv2d", "conv2d", {3, 3, 1, 1, 0, 0, 1, 1, 1}, {input_node_0, weight_node});

        conv_node->set_op_attr("channel_last", false);
        conv_node->set_op_attr("padding", std::vector<int>{1, 1, 1, 1});
        conv_node->set_op_attr("stride", std::vector<int>{1, 1});
        conv_node->set_op_attr("dilation", std::vector<int>{1, 1});

        create_output(*graph, "out", conv_node);
    }
};

struct UpdateReduceSumAttrsTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *reduce_node;

    UpdateReduceSumAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateReduceSumAttrsTest");
    }

   protected:
    graphlib::OpNode *create_graph(
        const std::string &reduce_op, int reduce_dim, bool keep_dim, const graphlib::Shape &input_shape)
    {
        auto input_node = create_input(*graph, "input", input_shape);

        reduce_node = add_node<graphlib::PyOpNode>(
            *graph,
            reduce_op,
            graphlib::OpType(reduce_op, {}, {{"dim_arg", std::vector<int>{reduce_dim}}, {"keep_dim", keep_dim}}),
            {input_node});
        create_output(*graph, "out", reduce_node);

        return reduce_node;
    }
};

TEST_F(UpdateReduceSumAttrsTest, ReduceSumDim)
{
    std::string reduce_op = "reduce_sum";
    int reduce_dim = 1;
    bool keep_dim = true;
    graphlib::Shape input_shape = graphlib::Shape::create({1, 512, 160});
    graphlib::Shape expected_shape = graphlib::Shape::create({1, 1, 160});

    auto reduce_node = create_graph(reduce_op, reduce_dim, keep_dim, input_shape);

    passes::update_reduce_attr(reduce_node, reduce_dim, keep_dim);

    auto updated_attrs = reduce_node->op_named_attrs();

    ASSERT_TRUE(updated_attrs.count("dim_arg"));
    auto dim_arg_vec = std::get<std::vector<int>>(updated_attrs["dim_arg"]);
    EXPECT_EQ(dim_arg_vec[0], reduce_dim);

    ASSERT_TRUE(updated_attrs.count("keep_dim"));
    EXPECT_EQ(std::get<bool>(updated_attrs["keep_dim"]), keep_dim);
}

struct UpdateReduceMaxAttrsTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *reduce_node;

    UpdateReduceMaxAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateReduceMaxAttrsTest");
    }

   protected:
    graphlib::OpNode *create_graph(
        const std::string &reduce_op, int reduce_dim, bool keep_dim, const graphlib::Shape &input_shape)
    {
        auto input_node = create_input(*graph, "input", input_shape);

        reduce_node = add_node<graphlib::PyOpNode>(
            *graph,
            reduce_op,
            graphlib::OpType(reduce_op, {}, {{"dim_arg", std::vector<int>{reduce_dim}}, {"keep_dim", keep_dim}}),
            {input_node});

        create_output(*graph, "out", reduce_node);
        return reduce_node;
    }
};

TEST_F(UpdateReduceMaxAttrsTest, ReduceMaxDim)
{
    std::string reduce_op = "reduce_max";
    int reduce_dim = 2;
    bool keep_dim = true;
    graphlib::Shape input_shape = graphlib::Shape::create({1, 512, 160});
    graphlib::Shape expected_shape = graphlib::Shape::create({1, 512, 1});

    auto reduce_node = create_graph(reduce_op, reduce_dim, keep_dim, input_shape);

    passes::update_reduce_attr(reduce_node, reduce_dim, keep_dim);

    auto updated_attrs = reduce_node->op_named_attrs();

    ASSERT_TRUE(updated_attrs.count("dim_arg"));
    auto dim_arg_vec = std::get<std::vector<int>>(updated_attrs["dim_arg"]);
    EXPECT_EQ(dim_arg_vec[0], reduce_dim);

    ASSERT_TRUE(updated_attrs.count("keep_dim"));
    EXPECT_EQ(std::get<bool>(updated_attrs["keep_dim"]), keep_dim);
}

struct EraseInverseOpsSqueezeAndUnsqueeze : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *squeeze_node;

    EraseInverseOpsSqueezeAndUnsqueeze()
    {
        graphlib::Shape mask_shape = graphlib::Shape::create({1, 1, 256, 256});
        graphlib::Shape weights_shape = graphlib::Shape::create({16, 256, 256});

        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "EraseInverseOpsSqueezeAndUnsqueezeTest");

        auto mask_node = create_input(*graph, "attention_mask", mask_shape);
        auto weights_node = create_input(*graph, "attention_weights", weights_shape);

        auto cast_1_node = add_node<graphlib::PyOpNode>(
            *graph,
            "cast",
            graphlib::OpType("cast", {}, {{"dtype", static_cast<int>(DataFormat::Float32)}}),
            {mask_node});
        auto unsqueeze_node = add_node<graphlib::PyOpNode>(
            *graph, "unsqueeze", graphlib::OpType("unsqueeze", {0}, {{"dim", 0}}), {weights_node});

        tt::graphlib::InputNode *maximum_input_1 =
            create_input(*graph, "input_1_maximum", graphlib::Shape::create({1}));
        auto add_1_node = add_node<graphlib::PyOpNode>(*graph, "add", "add", {}, {cast_1_node, unsqueeze_node});

        auto maximum_node =
            add_node<graphlib::PyOpNode>(*graph, "maximum", "maximum", {}, {maximum_input_1, add_1_node});

        squeeze_node = add_node<graphlib::PyOpNode>(
            *graph, "squeeze", graphlib::OpType("squeeze", {0}, {{"dim", 0}}), {maximum_node});

        create_output(*graph, "out", squeeze_node);
    }
};

TEST_F(EraseInverseOpsSqueezeAndUnsqueeze, erase_inv_ops_sq_unsq)
{
    reportify::dump_graph(graph->name(), "BEFORE_erase_inverse_ops", graph);
    bool erased = passes::erase_inverse_ops(graph);
    reportify::dump_graph(graph->name(), "AFTER_erase_inverse_ops", graph);
    EXPECT_TRUE(erased);

    std::vector<Node *> nodes = graphlib::topological_sort(*graph);
    Node *squeeze_node = nodes[4];
    graphlib::OpNode *squeeze_op = nodes[4]->as<graphlib::PyOpNode>();
    ASSERT_EQ(squeeze_op->new_op_type(), ops::OpType::Squeeze);
    graphlib::Node *operand_node = graph->operands(squeeze_node)[0];

    // Check that dimension on which we squeeze is 0
    auto reshape_attrs = squeeze_op->op_named_attrs();
    ASSERT_TRUE(reshape_attrs.count("dim"));

    int dim = std::get<int>(reshape_attrs["dim"]);
    EXPECT_EQ(dim, 0);
    graphlib::Shape shape = squeeze_op->shape();
    graphlib::Shape shape_of_operand = squeeze_op->shape_of_operand(graph, operand_node, /*ignore_broadcasts*/ true);
    // Method shape_of_operand takes into account any tms on input edges
    // We want this, since we want to check that input volume to the squeeze node is the same as the output volume
    EXPECT_EQ(shape.volume(), shape_of_operand.volume());
}

struct CommuteTransposeThroughReduce : testing::Test
{
    graphlib::Graph *graph;
    graphlib::Shape input_shape;

    CommuteTransposeThroughReduce()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "CommuteTransposeThroughReduce");

        // Create a 3D input tensor [32, 64, 128]
        input_shape = graphlib::Shape::create({32, 64, 128});
        auto input_node = create_input(*graph, "input", input_shape);

        // Add a transpose that swaps the first two dimensions (after which dims are [64, 32, 128])
        auto transpose_node = add_node<graphlib::PyOpNode>(
            *graph, "transpose", graphlib::OpType("transpose", {}, {{"dim0", 0}, {"dim1", 1}}), {input_node});

        // Add a reshape to increase dimensionality (3D -> 5D)
        // Reshape to [64, 4, 8, 16, 8] (same volume as [64, 32, 128])
        auto reshape_node = add_node<graphlib::PyOpNode>(
            *graph,
            "reshape",
            graphlib::OpType("reshape", {}, {{"shape", std::vector{64, 4, 8, 16, 8}}}),
            {transpose_node});

        // Add a reduce_avg on the last dimension (-1)
        auto reduce_node = add_node<graphlib::PyOpNode>(
            *graph,
            "reduce",
            graphlib::OpType("reduce_avg", {}, {{"dim_arg", std::vector<int>{-1}}, {"keep_dim", true}}),
            {reshape_node});

        create_output(*graph, "out", reduce_node);
    }
};

TEST_F(CommuteTransposeThroughReduce, commute_transpose_through_higher_dim_reduce)
{
    graphlib::OpNode *transpose_op = dynamic_cast<graphlib::OpNode *>(graph->get_node_by_name("transpose"));
    graphlib::OpNode *reduce_op = dynamic_cast<graphlib::OpNode *>(graph->get_node_by_name("reduce"));

    // shape before transposing
    graphlib::Shape commute_shape = input_shape;

    // shape after transposing
    graphlib::Shape clone_shape = transpose_op->shape();

    bool result = passes::can_commute_through_reduce(
        graph, reduce_op, transpose_op, nullptr, &commute_shape, &clone_shape, false);

    // we are expecting false here since transpose has less dimensions than reduce
    // so we can't commute transpose through reduce
    EXPECT_FALSE(result);
}
