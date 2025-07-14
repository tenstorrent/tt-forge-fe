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

namespace tt::test::ops::eltwise_binary
{

struct SimpleEltwiseBinaryTest : public tt::test::ops::BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    SimpleEltwiseBinaryTest() : BaseOpTest(GetParam()) {}
};

test::ops::eval_function_t get_golden_eval_function(const tt::ops::OpType& op_type)
{
    switch (op_type)
    {
        case tt::ops::OpType::Add:
            return [](const std::vector<torch::Tensor>& inputs) { return inputs[0] + inputs[1]; };
        case tt::ops::OpType::Multiply:
            return [](const std::vector<torch::Tensor>& inputs) { return inputs[0] * inputs[1]; };
        case tt::ops::OpType::Divide:
            return [](const std::vector<torch::Tensor>& inputs) { return inputs[0] / inputs[1]; };
        case tt::ops::OpType::Subtract:
            return [](const std::vector<torch::Tensor>& inputs) { return inputs[0] - inputs[1]; };
        default: throw std::runtime_error("Unsupported operation type for golden evaluation");
    }
}

TEST_P(SimpleEltwiseBinaryTest, test_eltwise_binary_eval)
{
    // Evaluate the graph node by node.
    eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    auto golden_output = golden_eval(input_tensors);

    verify_fwd(golden_eval);
}

TEST_P(SimpleEltwiseBinaryTest, test_eltwise_binary_decompose)
{
    // TODO: decomposing context needs `compiler_cfg`; passing nullptr for now...
    tt::decompose_tt_forge_graph<DecomposeEpoch::Initial>(get_graph(), std::shared_ptr<void>(nullptr, [](void*) {}));

    // Evaluate the graph node by node.
    eval_graph();

    // Compute golden output.
    auto input_tensors = get_input_tensors();
    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    auto golden_output = golden_eval(input_tensors);

    verify_fwd(golden_eval);
}

TEST_P(SimpleEltwiseBinaryTest, test_eltwise_binary_backward)
{
    run_autograd();

    // Verify the forward path.
    eval_graph();
    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    auto golden_output = golden_eval(get_input_tensors());
    verify_fwd(golden_eval);

    eval_graph(tt::graphlib::NodeEpochType::Backward);
    auto tensors = get_output_tensors_with_grads();
    {
        pybind11::gil_scoped_release gil_release;
        golden_output.backward(generated_grad());
    }

    verify_bwd();
}

INSTANTIATE_TEST_SUITE_P(
    BinaryOpsIndividual,
    SimpleEltwiseBinaryTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(
                tt::ops::OpType::Add, tt::ops::OpType::Multiply, tt::ops::OpType::Divide, tt::ops::OpType::Subtract),
            testing::Values(
                VecShapes{{1, 1, 1, 32}, {1, 1, 1, 32}},
                VecShapes{{1, 1, 32, 1}, {1, 1, 32, 1}},
                VecShapes{{1, 32, 1, 1}, {1, 32, 1, 1}},
                VecShapes{{32, 1, 1, 1}, {32, 1, 1, 1}},
                VecShapes{{1, 2, 3, 4}, {1, 2, 3, 4}},
                VecShapes{{2, 3, 4}, {2, 3, 4}},
                VecShapes{{3, 4}, {3, 4}},
                VecShapes{{4}, {4}},
                VecShapes{{1}, {1}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleEltwiseBinaryTest::ParamType>& info)
    {
        // Generate a test name based on the operation type and input shapes.
        const auto& param = info.param;
        static size_t id = 0;
        std::string op_name = param.op.as_string() + std::to_string(id++);
        return op_name;
    });

// Note: STANDARD_RANGE_OP_TEST_SET is designed for unary ops and cannot be used for binary ops
// as it generates only a single shape, but binary ops need two input shapes

}  // namespace tt::test::ops::eltwise_binary
//
namespace tt::test::ops::eltwise_unary
{

struct SimpleEltwiseUnary : public tt::test::ops::BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    SimpleEltwiseUnary() : BaseOpTest(GetParam()) {}
};

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
    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    verify_fwd(golden_eval);
}

TEST_P(SimpleEltwiseUnary, test_eltwise_unary_decompose)
{
    // TODO: decomposing context needs `compiler_cfg`; passing nullptr for now...
    tt::decompose_tt_forge_graph<DecomposeEpoch::Initial>(get_graph(), std::shared_ptr<void>(nullptr, [](void*) {}));

    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    verify_fwd(golden_eval);
}

TEST_P(SimpleEltwiseUnary, test_eltwise_unary_backward)
{
    run_autograd();

    auto golden_eval = get_golden_eval_function(GetParam().op.type());
    verify_fwd(golden_eval);

    auto golden_output = get_fwd_output_tensor();

    eval_graph(tt::graphlib::NodeEpochType::Backward);
    {
        pybind11::gil_scoped_release gil_release;
        golden_output.backward(generated_grad());
    }

    verify_bwd();
}

void test_range()
{
    // Test shape_range with different dimensions
    auto val1 = shape_range(std::array<uint32_t, 1>{1}, std::array<uint32_t, 1>{3});
    auto val2 = shape_range(std::array<uint32_t, 2>{1, 5}, std::array<uint32_t, 2>{3, 7});
    auto val3 = shape_range(std::array<uint32_t, 3>{1, 1, 1}, std::array<uint32_t, 3>{3, 7, 5});
    auto val4 = shape_range(std::array<uint32_t, 4>{1, 1, 1, 1}, std::array<uint32_t, 4>{3, 7, 5, 9});
}

void ff();
using ShapesInit = std::initializer_list<std::initializer_list<uint32_t>>;

using VecShapes = std::vector<graphlib::Shape>;

auto shape_val(const std::initializer_list<std::initializer_list<uint32_t>> shapes)
{
    auto vec_in = std::vector(shapes);
    return std::vector<graphlib::Shape>(vec_in.begin(), vec_in.end());
}

auto shape_values(std::initializer_list<ShapesInit> args)
{
    // return testing::Values(shape_val(args));
}

// Macro `SHAPE_VALUES` to simplify the creation of shape values.
// Forwards the arguments to the shape_values() function
// but explicitly casts each of the arguments to `std::initializer_list<std::initializer_list<uint32_t>>`
// to avoid issues with template type deduction.
#define SHAPE_VALUES(...) shape_values(std::initializer_list<ShapeInit>(__VA_ARGS__));

#define WRAP_ARGS(arg, ...) \
    std::initializer_list<std::initializer_list<uint32_t>>(arg) __VA_OPT__(, ) WRAP_ARGS(__VA_ARGS__)

void test_shape_vals()
{
    // auto vals = SHAPE_VALUES(
    //     {{1, 1, 2, 2}},
    //     {{1, 2, 3, 4}},
    //     {{1, 3, 5, 7}});
}

// INSTANTIATE_TEST_SUITE_P(
//     AbsOp,
//     SimpleEltwiseUnary,
//     testing::Combine(testing::Values(tt::ops::OpType::Abs), shape_values({{1, 1, 2, 2}}, {{1, 2, 3, 4}})));
//
// testing::Values(
//     OpTestParam{.op = tt::ops::OpType::Abs, .input_shapes = {{1, 1, 2, 2}}},
//     OpTestParam{.op = tt::ops::OpType::Abs, .input_shapes = {{1, 2, 3, 4}}}));

class TestStruct
{
    int op;
    std::string input_shapes;

   public:
    TestStruct(const std::tuple<int, std::string>& params) : op(std::get<0>(params)), input_shapes(std::get<1>(params))
    {
    }
};

class SimpleTest : public testing::WithParamInterface<TestStruct>
{
};

INSTANTIATE_TEST_SUITE_P(
    AbsOpIndividual,
    SimpleEltwiseUnary,
    testing::ConvertGenerator(
        testing::Combine(
            testing::Values(tt::ops::OpType::Abs),
            testing::Values(
                VecShapes{{1, 1, 1, 32}},
                VecShapes{{1, 1, 32, 1}},
                VecShapes{{1, 32, 1, 1}},
                VecShapes{{32, 1, 1, 1}},
                VecShapes{{1, 2, 3, 4}},
                VecShapes{{2, 3, 4}},
                VecShapes{{3, 4}},
                VecShapes{{4}},
                VecShapes{{1}})),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

STANDARD_RANGE_OP_TEST_SET(Abs, tt::ops::OpType::Abs, SimpleEltwiseUnary);

}  // namespace tt::test::ops::eltwise_unary
