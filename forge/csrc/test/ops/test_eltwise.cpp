// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/python_bindings.hpp"
#include "ops/op.hpp"
#include "passes/decomposing_context.hpp"
#include "pybind11/gil.h"
#include "test/common.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::eltwise_binary
{

struct SimpleEltwiseBinaryTest : public tt::test::ops::BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    SimpleEltwiseBinaryTest() : BaseOpTest(GetParam()) {}
};

// TEST_P(SimpleEltwiseBinaryTest, test_eltwise_binary_shape)
// {
//     graphlib::Graph* graph = get_graph();
//
//     EXPECT_EQ(tt::eval_graph_simple(graph).sizes(), (m_in1 + m_in2).sizes());
// }
//
//

test::ops::eval_function_t get_golden_eval_function(const tt::ops::OpType& op_type)
{
    switch (op_type)
    {
        case tt::ops::OpType::Add:
            return [](const std::vector<torch::Tensor>& inputs) { return inputs[0] + inputs[1]; };
        case tt::ops::OpType::Multiply:
            return [](const std::vector<torch::Tensor>& inputs) { return inputs[0] * inputs[1]; };
        // Add more cases for other binary operations as needed
        default: throw std::runtime_error("Unsupported operation type for golden evaluation");
    }
}

// TEST_P(SimpleEltwiseBinaryTest, test_eltwise_binary_eval)
// {
//     // Evaluate the graph using the golden evaluation function
//     auto golden_eval = get_golden_eval_function(GetParam().op);
//     eval_graph(golden_eval);
// }
//
// INSTANTIATE_TEST_SUITE_P(
//     BinaryOps,
//     SimpleEltwiseBinaryTest,
//     testing::Values(
//         OpTestParam{.op = tt::ops::OpType::Add, .input_shapes = {{1, 1, 2, 2}, {1, 1, 2, 2}}},
//         OpTestParam{.op = tt::ops::OpType::Multiply, .input_shapes = {{1, 1, 2, 2}, {1, 1, 2, 2}}}));

}  // namespace tt::test::ops::eltwise_binary
//
namespace tt::test::ops::eltwise_unary
{

struct SimpleEltwiseUnary : public tt::test::ops::BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    SimpleEltwiseUnary() : BaseOpTest(GetParam()) {}
};

// TEST_P(SimpleEltwiseBinaryTest, test_eltwise_binary_shape)
// {
//     graphlib::Graph* graph = get_graph();
//
//     EXPECT_EQ(tt::eval_graph_simple(graph).sizes(), (m_in1 + m_in2).sizes());
// }
//
//

test::ops::eval_function_t get_golden_eval_function(const tt::ops::OpType& op_type)
{
    switch (op_type)
    {
        case tt::ops::OpType::Abs:
            return [](const std::vector<torch::Tensor>& inputs) { return torch::abs(inputs[0]); };
        // case tt::ops::OpType::Multiply:
        //     return [](const std::vector<torch::Tensor>& inputs) { return inputs[0] * inputs[1]; };
        // Add more cases for other binary operations as needed
        default: throw std::runtime_error("Unsupported operation type for golden evaluation");
    }
}

TEST_P(SimpleEltwiseUnary, test_eltwise_unary_eval)
{
    // Evaluate the graph node by node.
    auto output = eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    auto golden_output = golden_eval(input_tensors);

    assert_equal(golden_output, output);
}

TEST_P(SimpleEltwiseUnary, test_eltwise_unary_decompose)
{
    // TODO: decomposing context needs `compiler_cfg`; passing nullptr for now...
    tt::decompose_tt_forge_graph<DecomposeEpoch::Initial>(get_graph(), std::shared_ptr<void>(nullptr, [](void*) {}));

    // Evaluate the graph node by node.
    auto output = eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    auto golden_output = golden_eval(input_tensors);

    assert_equal(golden_output, output);
}

TEST_P(SimpleEltwiseUnary, test_eltwise_unary_backward)
{
    run_autograd();

    // Verify the forward path.
    auto output = eval_graph();
    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    auto golden_output = golden_eval(get_input_tensors());
    assert_equal(golden_output, output);

    auto bwd_output = eval_graph(tt::graphlib::NodeEpochType::Backward);
    {
        pybind11::gil_scoped_release gil_release;
        golden_output.backward(generated_grad());
    }

    assert_equal(get_input_tensors()[0].grad(), bwd_output);
}

INSTANTIATE_TEST_SUITE_P(
    BinaryOps,
    SimpleEltwiseUnary,
    testing::Values(OpTestParam{.op = tt::ops::OpType::Abs, .input_shapes = {{1, 1, 2, 2}}}));

}  // namespace tt::test::ops::eltwise_unary
