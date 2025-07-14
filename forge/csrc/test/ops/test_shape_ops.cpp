// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gtest/internal/gtest-param-util.h>
#include <torch/torch.h>

#include <array>
#include <initializer_list>
#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/python_bindings.hpp"
#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "ops/op.hpp"
#include "passes/decomposing_context.hpp"
#include "pybind11/gil.h"
#include "test/common.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::shape_ops
{

struct ReshapeTest : public tt::test::ops::BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    ReshapeTest() : BaseOpTest(GetParam()) {}
};

struct TransposeTest : public tt::test::ops::BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    TransposeTest() : BaseOpTest(GetParam()) {}
};

test::ops::eval_function_t get_golden_eval_function(const tt::ops::OpType& op_type, const tt::ops::Op& op)
{
    switch (op_type)
    {
        case tt::ops::OpType::Reshape:
        {
            const std::vector<int>& shape_vec = op.attr_as<std::vector<int>>("shape");
            std::vector<long> target_shape(shape_vec.begin(), shape_vec.end());
            return [target_shape](const std::vector<torch::Tensor>& inputs) { return inputs[0].reshape(target_shape); };
        }
        case tt::ops::OpType::Transpose:
        {
            int d0 = op.attr_as<int>("dim0");
            int d1 = op.attr_as<int>("dim1");
            return [d0, d1](const std::vector<torch::Tensor>& inputs) { return inputs[0].transpose(d0, d1); };
        }
        default: throw std::runtime_error("Unsupported operation type for golden evaluation");
    }
}

TEST_P(ReshapeTest, test_reshape_eval)
{
    // Evaluate the graph node by node.
    eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type(), GetParam().op);
    auto golden_output = golden_eval(input_tensors);

    assert_equal(golden_output, get_fwd_output_tensor());
}

TEST_P(ReshapeTest, test_reshape_decompose)
{
    // TODO: decomposing context needs `compiler_cfg`; passing nullptr for now...
    tt::decompose_tt_forge_graph<DecomposeEpoch::Initial>(get_graph(), std::shared_ptr<void>(nullptr, [](void*) {}));

    // Evaluate the graph node by node.
    eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type(), GetParam().op);
    auto golden_output = golden_eval(input_tensors);

    assert_equal(golden_output, get_fwd_output_tensor());
}

TEST_P(ReshapeTest, test_reshape_backward)
{
    run_autograd();

    // Verify the forward path.
    eval_graph();
    auto golden_eval = get_golden_eval_function(GetParam().op.type(), GetParam().op);
    auto golden_output = golden_eval(get_input_tensors());
    assert_equal(golden_output, get_fwd_output_tensor());

    eval_graph(tt::graphlib::NodeEpochType::Backward);
    auto tensors = get_output_tensors_with_grads();
    {
        pybind11::gil_scoped_release gil_release;
        golden_output.backward(generated_grad());
    }

    for (const auto& tensor : tensors)
    {
        assert_equal(tensor.second, tensor.first.grad());
    }
}

TEST_P(TransposeTest, test_transpose_eval)
{
    // Evaluate the graph node by node.
    eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type(), GetParam().op);
    auto golden_output = golden_eval(input_tensors);

    assert_equal(golden_output, get_fwd_output_tensor());
}

TEST_P(TransposeTest, test_transpose_decompose)
{
    // TODO: decomposing context needs `compiler_cfg`; passing nullptr for now...
    tt::decompose_tt_forge_graph<DecomposeEpoch::Initial>(get_graph(), std::shared_ptr<void>(nullptr, [](void*) {}));

    // Evaluate the graph node by node.
    eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type(), GetParam().op);
    auto golden_output = golden_eval(input_tensors);

    assert_equal(golden_output, get_fwd_output_tensor());
}

TEST_P(TransposeTest, test_transpose_backward)
{
    run_autograd();

    // Verify the forward path.
    eval_graph();
    auto golden_eval = get_golden_eval_function(GetParam().op.type(), GetParam().op);
    auto golden_output = golden_eval(get_input_tensors());
    assert_equal(golden_output, get_fwd_output_tensor());

    eval_graph(tt::graphlib::NodeEpochType::Backward);
    auto tensors = get_output_tensors_with_grads();
    {
        pybind11::gil_scoped_release gil_release;
        golden_output.backward(generated_grad());
    }

    for (const auto& tensor : tensors)
    {
        assert_equal(tensor.second, tensor.first.grad());
    }
}

// Helper function to create Reshape ops with shape attribute
tt::ops::Op create_reshape_op(const std::vector<int>& target_shape)
{
    tt::ops::Op op(tt::ops::OpType::Reshape);
    op.set_attr("shape", target_shape);
    return op;
}

// Helper function to create Transpose ops with dim attributes
tt::ops::Op create_transpose_op(int dim0, int dim1)
{
    tt::ops::Op op(tt::ops::OpType::Transpose);
    op.set_attr("dim0", dim0);
    op.set_attr("dim1", dim1);
    return op;
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeOps,
    ReshapeTest,
    testing::ConvertGenerator(
        testing::Values(
            std::make_tuple(create_reshape_op({1, 1, 1, 24}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 1, 24, 1}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 24, 1, 1}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 2, 3, 4}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 4, 3, 2}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({2, 3, 4}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({6, 4}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({24}), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_reshape_op({1, 1, 6, 4}), VecShapes{{2, 12}}),
            std::make_tuple(create_reshape_op({1, 24}), VecShapes{{2, 12}}),
            std::make_tuple(create_reshape_op({24, 1}), VecShapes{{24, 1}}),
            std::make_tuple(create_reshape_op({1, 1, 24, 1}), VecShapes{{24, 1}}),
            std::make_tuple(create_reshape_op({8, 3}), VecShapes{{24, 1}}),
            std::make_tuple(create_reshape_op({1, 2, 2, 6}), VecShapes{{2, 2, 2, 3}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

INSTANTIATE_TEST_SUITE_P(
    TransposeOps,
    TransposeTest,
    testing::ConvertGenerator(
        testing::Values(
            // 4D tensors
            std::make_tuple(create_transpose_op(0, 1), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(0, 2), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(0, 3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(1, 2), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(1, 3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(2, 3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(-1, -2), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(-2, -3), VecShapes{{1, 2, 3, 4}}),
            std::make_tuple(create_transpose_op(-3, -4), VecShapes{{1, 2, 3, 4}}),
            // 3D tensors
            std::make_tuple(create_transpose_op(0, 1), VecShapes{{5, 4, 3}}),
            std::make_tuple(create_transpose_op(0, 2), VecShapes{{5, 4, 3}}),
            std::make_tuple(create_transpose_op(1, 2), VecShapes{{5, 4, 3}}),
            std::make_tuple(create_transpose_op(-1, -2), VecShapes{{5, 4, 3}}),
            // 2D tensors
            std::make_tuple(create_transpose_op(0, 1), VecShapes{{3, 4}}),
            std::make_tuple(create_transpose_op(-1, -2), VecShapes{{3, 4}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

// Range tests for Reshape
INSTANTIATE_TEST_SUITE_P(
    ReshapeRange2D,
    ReshapeTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(
                create_reshape_op({1, 1, 1, 20}),
                create_reshape_op({1, 1, 20, 1}),
                create_reshape_op({1, 20, 1, 1}),
                create_reshape_op({20, 1, 1, 1}),
                create_reshape_op({2, 10}),
                create_reshape_op({4, 5}),
                create_reshape_op({5, 4}),
                create_reshape_op({20})),
            testing::Values(VecShapes{{4, 5}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

// Note: STANDARD_RANGE_OP_TEST_SET generates shapes with varying dimensions,
// but transpose dims must be valid for the generated shape dimensions

}  // namespace tt::test::ops::shape_ops
