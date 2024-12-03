// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/edge.hpp"
#include "graph_lib/node_types.hpp"
#include "gtest/gtest.h"
#include "passes/commute_utils.hpp"
#include "passes/erase_inverse_ops.hpp"
#include "passes/insert_inverse_on_io.hpp"
#include "passes/replace_incommutable_patterns.hpp"
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
            *graph, "transpose0", graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}}), {matmul0});

        auto in1_a = create_input(*graph, "in1_a", graphlib::Shape::create({1, 1, shape[2], 128}));
        auto in1_b = create_input(*graph, "in1_b", graphlib::Shape::create({1, 1, 128, shape[3]}));
        auto matmul1 = add_node<graphlib::PyOpNode>(*graph, "matmul1", "matmul", {}, {in1_a, in1_b});
        auto transpose1 = add_node<graphlib::PyOpNode>(
            *graph, "transpose1", graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}}), {matmul1});

        auto add = add_node<graphlib::PyOpNode>(*graph, "add", "add", {}, {transpose0, transpose1});
        auto post_transpose = add_node<graphlib::PyOpNode>(
            *graph, "post_transpose", graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}}), {add});

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
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
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
            if (node->as<graphlib::PyOpNode>()->op_type().op == "transpose")
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
        *graph, "unary_transpose", graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}}), {unary});
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
            if (node->as<graphlib::PyOpNode>()->op_type().op == "transpose")
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
    auto smx_1 = add_node<graphlib::PyOpNode>(*graph, "smx_1", "softmax", {-1, false}, {post_transpose});
    auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 512, 10, 16}, {smx_1});
    auto reduce_1 = add_node<graphlib::PyOpNode>(*graph, "reduce_1", "reduce_sum", {-2, true}, {reshape_1});
    auto reduce_2 = add_node<graphlib::PyOpNode>(*graph, "reduce_2", "reduce_sum", {-1, true}, {reduce_1});
    auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {1, 1, 512, 1}, {reduce_2});
    auto smx_2 = add_node<graphlib::PyOpNode>(*graph, "smx_2", "softmax", {-1, false}, {reshape_2});
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
            if (node->as<graphlib::PyOpNode>()->op_type().op == "transpose")
                transpose_count++;
            else if (node->as<graphlib::PyOpNode>()->op_type().op == "reduce_sum")
                reduce_count++;
            else if (node->as<graphlib::PyOpNode>()->op_type().op == "reshape")
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
    auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {256, 320}, {post_transpose});
    auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {256, 320}, {post_transpose});
    auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {256, 320}, {post_transpose});
    auto concat =
        add_node<graphlib::PyOpNode>(*graph, "concat", "concatenate", {-2}, {reshape_0, reshape_1, reshape_2});

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
            if (node->as<graphlib::PyOpNode>()->op_type().op == "reshape")
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
            *graph, "transpose", graphlib::OpType("transpose", {}, {}, {{"dim0", -3}, {"dim1", -2}}), {in0_a});

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
    EXPECT_EQ(tms[0].op, "broadcast");
    EXPECT_EQ(tms[1].op, "broadcast");
    // Broadcast along dimension -2 should become broadcast along -3
    // after commuting through transpose.
    EXPECT_EQ(std::get<int>(tms[1].attr[0]), -3);
    EXPECT_EQ(std::get<int>(tms[1].attr[1]), 112);
}

struct UpdateReshapeNamedAttrsTest : testing::Test
{
    graphlib::Graph *graph;

    UpdateReshapeNamedAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateReshapeNamedAttrs");

        graphlib::Shape initial_shape = graphlib::Shape::create({1, 512, 160});
        tt::graphlib::InputNode *input_node = create_input(*graph, "input", initial_shape);

        auto reshape_node = add_node<graphlib::PyOpNode>(*graph, "reshape", "reshape", {1, 1, 512 * 160}, {input_node});

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

    auto updated_attrs = op_node_reshape->op_type().named_attrs;
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
        tt::graphlib::InputNode *input_node = create_input(*graph, "input", initial_shape);
        auto select_node =
            add_node<graphlib::PyOpNode>(*graph, "select", "select", {dim, begin, length, stride}, {input_node});

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

    auto updated_attrs = op_node_select->named_attrs();

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

        reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {1, 1, 512 * 160}, {input_node_0});
        reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 1, 512 * 160}, {input_node_1});
        reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {1, 1, 512 * 160}, {input_node_2});

        auto concat_node =
            add_node<graphlib::PyOpNode>(*graph, "concat", "concatenate", {-2}, {reshape_0, reshape_1, reshape_2});

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
    auto updated_attrs = op_node_concat->named_attrs();
    EXPECT_TRUE(updated_attrs.count("dim")) << "Dim attribute not found.";
    auto dim_value = std::get<int>(updated_attrs["dim"]);
    EXPECT_EQ(dim_value, new_dim) << "Dim attribute does not match expected value.";
}

struct UpdateVStackAttrsTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *vstack_node;

    UpdateVStackAttrsTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateVStackAttrs");

        graphlib::Shape shape_0 = graphlib::Shape::create({32, 512, 160});

        auto input_node_0 = create_input(*graph, "input_0", shape_0);

        vstack_node = add_node<graphlib::PyOpNode>(*graph, "vstack", "vstack", {16}, {input_node_0});

        create_output(*graph, "out", vstack_node);
    }
};

TEST_F(UpdateVStackAttrsTest, update_vstack_attr)
{
    graphlib::Node *vstack = graph->get_node_by_name("vstack");
    ASSERT_NE(vstack, nullptr) << "VStack node not found.";
    graphlib::OpNode *op_node_vstack = dynamic_cast<graphlib::OpNode *>(vstack);
    ASSERT_NE(op_node_vstack, nullptr) << "Node is not of type OpNode.";

    int new_slice_size = 32;

    passes::update_vstack_attr(op_node_vstack, new_slice_size);

    auto updated_attrs = op_node_vstack->named_attrs();

    EXPECT_TRUE(updated_attrs.count("slice_size")) << "Slice size attribute not found.";
    auto slice_size_value = std::get<int>(updated_attrs["slice_size"]);
    EXPECT_EQ(slice_size_value, new_slice_size) << "Slice size attribute does not match expected value.";
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

TEST_F(UpdateMatMulNamedAttrsTest, update_named_attrs)
{
    graphlib::Node *matmul_node = graph->get_node_by_name("matmul");
    ASSERT_NE(matmul_node, nullptr) << "MatMul node not found.";

    graphlib::OpNode *op_node_matmul = dynamic_cast<graphlib::OpNode *>(matmul_node);
    ASSERT_NE(op_node_matmul, nullptr) << "Node is not of type OpNode.";
    int requant_zp = 128;
    passes::update_matmul_attr(op_node_matmul, requant_zp);
    auto updated_attrs = op_node_matmul->named_attrs();
    EXPECT_TRUE(updated_attrs.count("requant_zp")) << "Requant zp attribute not found.";

    auto zp_value = std::get<int>(updated_attrs["requant_zp"]);
    EXPECT_EQ(zp_value, requant_zp) << "Requant zp attribute does not match expected value.";

    auto matmul_attrs = op_node_matmul->op_attrs();
    ASSERT_FALSE(matmul_attrs.empty()) << "MatMul attributes should not be empty.";
    EXPECT_EQ(std::get<int>(matmul_attrs.back()), requant_zp) << "Requant zp not correctly appended to op_attrs.";
}

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
        auto named_attrs = conv_node->named_attrs();
        named_attrs["channel_last"] = false;
        named_attrs["padding_top"] = 1;
        named_attrs["padding_bottom"] = 1;
        named_attrs["padding_left"] = 1;
        named_attrs["padding_right"] = 1;
        named_attrs["stride_height"] = 1;
        named_attrs["stride_width"] = 1;
        named_attrs["dilation_height"] = 1;
        named_attrs["dilation_width"] = 1;

        conv_node->overwrite_named_attrs(named_attrs);
        create_output(*graph, "out", conv_node);
    }
};

TEST_F(UpdateConvAttrsTest, update_padding_attrs)
{
    graphlib::Node *conv = graph->get_node_by_name("conv2d");
    ASSERT_NE(conv, nullptr) << "Conv2d node not found.";

    graphlib::OpNode *op_node_conv = dynamic_cast<graphlib::OpNode *>(conv);
    ASSERT_NE(op_node_conv, nullptr) << "Node is not of type OpNode.";

    std::vector<int> new_padding = {2, 3, 4, 5};

    passes::update_conv_attr(op_node_conv, new_padding);

    auto updated_attrs = op_node_conv->op_type().named_attrs;

    EXPECT_TRUE(updated_attrs.count("padding")) << "Padding attribute not found in named attributes.";
    auto updated_padding = std::get<std::vector<int>>(updated_attrs["padding"]);

    EXPECT_EQ(updated_padding, new_padding) << "Padding attribute does not match the expected values.";

    auto conv_attrs = op_node_conv->op_attrs();
    int pad_idx_offset = 4;
    for (size_t i = 0; i < 4; i++)
    {
        EXPECT_EQ(std::get<int>(conv_attrs[pad_idx_offset + i]), new_padding[i])
            << "Padding value at index " << i << " does not match the expected value.";
    }
}

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
            reduce_op,
            {
                reduce_dim,
                keep_dim,
            },
            {input_node});
        auto &named_attrs = reduce_node->named_attrs();
        named_attrs["dim"] = reduce_dim;
        named_attrs["keep_dim"] = keep_dim;
        reduce_node->overwrite_named_attrs(named_attrs);
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

    auto updated_attrs = reduce_node->named_attrs();

    ASSERT_TRUE(updated_attrs.count("dim"));
    EXPECT_EQ(std::get<int>(updated_attrs["dim"]), reduce_dim);

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
        const std::string &reduce_op, int reduce_dim, int stride, bool keep_dim, const graphlib::Shape &input_shape)
    {
        auto input_node = create_input(*graph, "input", input_shape);

        reduce_node = add_node<graphlib::PyOpNode>(
            *graph,
            reduce_op,
            reduce_op,
            {
                reduce_dim,
                stride,
                keep_dim,
            },
            {input_node});
        auto &named_attrs = reduce_node->named_attrs();
        named_attrs["dim"] = reduce_dim;
        named_attrs["stride"] = stride;
        named_attrs["keep_dim"] = keep_dim;
        reduce_node->overwrite_named_attrs(named_attrs);
        create_output(*graph, "out", reduce_node);

        return reduce_node;
    }
};

TEST_F(UpdateReduceMaxAttrsTest, ReduceMaxDim)
{
    std::string reduce_op = "reduce_max";
    int reduce_dim = 2;
    int stride = 1;
    bool keep_dim = true;
    graphlib::Shape input_shape = graphlib::Shape::create({1, 512, 160});
    graphlib::Shape expected_shape = graphlib::Shape::create({1, 512, 1});

    auto reduce_node = create_graph(reduce_op, reduce_dim, stride, keep_dim, input_shape);

    passes::update_reduce_attr(reduce_node, reduce_dim, keep_dim);

    auto updated_attrs = reduce_node->named_attrs();

    ASSERT_TRUE(updated_attrs.count("dim"));
    EXPECT_EQ(std::get<int>(updated_attrs["dim"]), reduce_dim);

    ASSERT_TRUE(updated_attrs.count("stride"));
    EXPECT_EQ(std::get<int>(updated_attrs["stride"]), stride);

    ASSERT_TRUE(updated_attrs.count("keep_dim"));
    EXPECT_EQ(std::get<bool>(updated_attrs["keep_dim"]), keep_dim);
}

struct UpdateGroupedReduceAvgTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *reduce_node;

    UpdateGroupedReduceAvgTest()
    {
        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "UpdateGroupedReduceAvg");
    }

   protected:
    graphlib::OpNode *create_graph(
        const std::string &reduce_op, const std::vector<int> &attr, const graphlib::Shape &input_shape)
    {
        auto input_node = create_input(*graph, "input", input_shape);
        reduce_node =
            add_node<graphlib::PyOpNode>(*graph, reduce_op, reduce_op, {attr[0], attr[1], attr[2]}, {input_node});
        auto &named_attrs = reduce_node->named_attrs();
        named_attrs["reduce_dim"] = attr[0];
        named_attrs["groups"] = attr[1];
        named_attrs["keep_dims"] = attr[2];
        reduce_node->overwrite_named_attrs(named_attrs);

        create_output(*graph, "out", reduce_node);

        return reduce_node;
    }
};

TEST_F(UpdateGroupedReduceAvgTest, GroupedReduceAvgDim)
{
    std::string reduce_op = "grouped_reduce_avg";
    std::vector<int> attr = {1, 4, 1};
    graphlib::Shape input_shape = graphlib::Shape::create({1, 512, 160});
    graphlib::Shape expected_shape = graphlib::Shape::create({1, 4, 160});

    auto reduce_node = create_graph(reduce_op, attr, input_shape);

    passes::update_grouped_reduce_avg_attr(reduce_node, attr[0]);

    auto updated_attrs = reduce_node->named_attrs();

    ASSERT_TRUE(updated_attrs.count("reduce_dim"));
    EXPECT_EQ(std::get<int>(updated_attrs["reduce_dim"]), attr[0]);
}

struct EraseInverseOpsSqueezeAndUnsqueezeTest : testing::Test
{
    graphlib::Graph *graph;
    graphlib::OpNode *squeeze_node;

    EraseInverseOpsSqueezeAndUnsqueezeTest()
    {
        graphlib::Shape mask_shape = graphlib::Shape::create({1, 1, 256, 256});
        graphlib::Shape weights_shape = graphlib::Shape::create({16, 256, 256});

        graph = new graphlib::Graph(graphlib::IRLevel::IR_TT_FORGE, "EraseInverseOpsSqueezeAndUnsqueezeTest");

        auto mask_node = create_input(*graph, "attention_mask", mask_shape);
        auto weights_node = create_input(*graph, "attention_weights", weights_shape);

        auto cast_1_node = add_node<graphlib::PyOpNode>(*graph, "cast", "cast", {"Float32"}, {mask_node});
        auto unsqueeze_node = add_node<graphlib::PyOpNode>(*graph, "unsqueeze", "unsqueeze", {0, 3}, {weights_node});

        tt::graphlib::InputNode *maximum_input_1 =
            create_input(*graph, "input_1_maximum", graphlib::Shape::create({1}));
        auto add_1_node = add_node<graphlib::PyOpNode>(*graph, "add", "add", {}, {cast_1_node, unsqueeze_node});

        auto maximum_node =
            add_node<graphlib::PyOpNode>(*graph, "maximum", "maximum", {}, {maximum_input_1, add_1_node});

        squeeze_node = add_node<graphlib::PyOpNode>(*graph, "squeeze", "squeeze", {0}, {maximum_node});

        create_output(*graph, "out", squeeze_node);
    }
};

TEST_F(EraseInverseOpsSqueezeAndUnsqueezeTest, EraseInverseOpsSqueezeAndUnsqueeze)
{
    bool erased = passes::erase_inverse_ops(graph);
    EXPECT_TRUE(erased);

    auto nodes = graphlib::topological_sort(*graph);

    graphlib::OpNode *squeeze_op = nodes[4]->as<graphlib::PyOpNode>();
    ASSERT_EQ(squeeze_op->op_name(), "squeeze");

    auto reshape_attrs = squeeze_op->named_attrs();
    ASSERT_TRUE(reshape_attrs.count("dim"));

    int dim = std::get<int>(reshape_attrs["dim"]);
    EXPECT_EQ(dim, 0);
}
