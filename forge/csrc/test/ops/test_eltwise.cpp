// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "lower_to_forge/common.hpp"
#include "ops/op.hpp"
#include "test/ops/test_ops.hpp"

namespace tt::test::ops::eltwise_binary
{

std::vector<tt::ops::Op> get_binary_eltwise_ops()
{
    return {
        tt::ops::OpType::Add,
        tt::ops::OpType::Multiply,
        tt::ops::OpType::Divide,
        tt::ops::OpType::Subtract,
        tt::ops::OpType::Power,
    };
}

std::vector<tt::ops::Op> get_binary_eltwise_ops_no_backward()
{
    return {
        tt::ops::OpType::Maximum,
        tt::ops::OpType::Minimum,
        tt::ops::OpType::Remainder,
        tt::ops::OpType::Heaviside,
    };
}

std::vector<VecShapes> get_binary_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 1, 32}, {1, 1, 1, 32}},
        VecShapes{{1, 1, 32, 1}, {1, 1, 32, 1}},
        VecShapes{{1, 32, 1, 1}, {1, 32, 1, 1}},
        VecShapes{{32, 1, 1, 1}, {32, 1, 1, 1}},
        VecShapes{{1, 2, 3, 4}, {1, 2, 3, 4}},
        VecShapes{{2, 3, 4}, {2, 3, 4}},
        VecShapes{{3, 4}, {3, 4}},
        VecShapes{{4}, {4}},
        VecShapes{{1}, {1}}};
}

bool valid_inputs(const graphlib::Shape& in1, const graphlib::Shape& in2)
{
    return in1 == in2 || graphlib::can_be_broadcasted(in1, in2);
}

std::vector<std::vector<graphlib::Shape>> generate_input_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range({1}, {16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {5, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1, 1}, {5, 1, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        input_shapes.push_back({shapes[i], shapes[i]});
        for (size_t j = i + 1; j < shapes.size(); ++j)
        {
            if (valid_inputs(shapes[i], shapes[j]))
            {
                input_shapes.push_back({shapes[i], shapes[j]});
                input_shapes.push_back({shapes[j], shapes[i]});
            }
        }
    }

    return input_shapes;
}

INSTANTIATE_TEST_SUITE_P(
    BinaryEltwiseOpsIndividual,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_binary_eltwise_ops()), testing::ValuesIn(get_binary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    BinaryEltwiseOpsSweep,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_binary_eltwise_ops()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

// Binary eltwise ops that don't support backward pass
INSTANTIATE_TEST_SUITE_P(
    BinaryEltwiseOpsNoBackwardIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_binary_eltwise_ops_no_backward()),
            testing::ValuesIn(get_binary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    BinaryEltwiseOpsNoBackwardSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_binary_eltwise_ops_no_backward()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::eltwise_binary

namespace tt::test::ops::eltwise_unary
{

std::vector<tt::ops::Op> get_unary_eltwise_ops()
{
    return {
        tt::ops::OpType::Abs,
        // tt::ops::OpType::Clip, // Tested separately.
        // tt::ops::OpType::CumulativeSum, // Tested separately.
        tt::ops::OpType::Sine,
        // tt::ops::OpType::Cast, // Tested separately.
        tt::ops::OpType::Cosine,
        tt::ops::OpType::Relu,
        tt::ops::OpType::Log,
        tt::ops::OpType::Sqrt,
        tt::ops::OpType::Tanh,
        tt::ops::OpType::Sigmoid,
        // tt::ops::OpType::Gelu,  // Has decompose bugs
        tt::ops::OpType::Exp,
        tt::ops::OpType::Reciprocal,
        tt::ops::OpType::Nop,
        // tt::ops::OpType::LeakyRelu,  // Has decompose bugs
        tt::ops::OpType::Erf,
        tt::ops::OpType::Tanh,
        // tt::ops::OpType::Pow, // Tested separately.
    };
}

std::vector<tt::ops::Op> get_unary_eltwise_ops_no_backward()
{
    return {
        tt::ops::OpType::Erf,
        tt::ops::OpType::Atan,
    };
}

std::vector<VecShapes> get_unary_individual_test_shapes()
{
    return {
        VecShapes{{1, 1, 1, 32}},
        VecShapes{{1, 1, 32, 1}},
        VecShapes{{1, 32, 1, 1}},
        VecShapes{{32, 1, 1, 1}},
        VecShapes{{1, 2, 3, 4}},
        VecShapes{{2, 3, 4}},
        VecShapes{{3, 4}},
        VecShapes{{4}},
        VecShapes{{1}}};
}

std::vector<std::vector<graphlib::Shape>> generate_input_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;

    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range({1}, {16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {16, 16});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {5, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1, 1}, {5, 1, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        input_shapes.push_back({shapes[i]});
    }

    return input_shapes;
}

INSTANTIATE_TEST_SUITE_P(
    UnaryEltwiseOpsIndividual,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_unary_eltwise_ops()), testing::ValuesIn(get_unary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    UnaryEltwiseOpsSweep,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_unary_eltwise_ops()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

// Unary eltwise ops that don't have backward pass impl.
INSTANTIATE_TEST_SUITE_P(
    UnaryEltwiseOpsNoBackwardIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_unary_eltwise_ops_no_backward()),
            testing::ValuesIn(get_unary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    UnaryEltwiseOpsNoBackwardSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_unary_eltwise_ops_no_backward()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

/**
 * Testing clip operation.
 */
std::vector<tt::ops::Op> get_clip_ops()
{
    std::vector<tt::ops::Op> ops;
    for (float min : {0.0f, 0.1f, 0.3f, 1.0f, 5.0f, 12.5f, 20.01f})
    {
        for (float max : {0.0f, 0.1f, 0.3f, 1.0f, 5.0f, 12.5f, 20.01f})
        {
            ops.push_back(tt::ops::Op(tt::ops::OpType::Clip, {{"min", min}, {"max", max}}));
        }
    }

    return ops;
}

INSTANTIATE_TEST_SUITE_P(
    ClipOpIndividual,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_clip_ops()), testing::ValuesIn(get_unary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    ClipOpSweeps,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_clip_ops()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

/**
 * Testing pow operation.
 */
std::vector<tt::ops::Op> get_pow_ops()
{
    std::vector<tt::ops::Op> ops;
    for (float exp : {0.0f, 0.1f, 2.0f, 3.0f, 3.5f, 12.5f, 20.01f})
        ops.push_back(tt::ops::Op(tt::ops::OpType::Pow, {{"exponent", exp}}));

    return ops;
}

INSTANTIATE_TEST_SUITE_P(
    PowOpIndividual,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_pow_ops()), testing::ValuesIn(get_unary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    PowOpSweeps,
    SimpleOpTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_pow_ops()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

/**
 * Testing cast operation.
 */
std::vector<tt::ops::Op> get_cast_ops()
{
    std::vector<tt::ops::Op> ops;
    for (const DataFormat& df : {
             DataFormat::Float32,
             DataFormat::Float16,
             DataFormat::Bfp8,
             DataFormat::Bfp4,
             DataFormat::Bfp2,
             DataFormat::Float16_b,
             DataFormat::Bfp8_b,
             DataFormat::Bfp4_b,
             DataFormat::RawUInt8,
             DataFormat::RawUInt32,
             DataFormat::UInt16,
             DataFormat::Int32,
         })
    {
        ops.push_back(tt::ops::Op(tt::ops::OpType::Cast, {{"dtype", static_cast<int>(df)}}));
    }

    return ops;
}

INSTANTIATE_TEST_SUITE_P(
    CastOpIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_cast_ops()), testing::ValuesIn(get_unary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    CastOpSweeps,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_cast_ops()), testing::ValuesIn(generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpTest::ParamType>& info) { return SimpleOpTest::get_test_name(info); });

}  // namespace tt::test::ops::eltwise_unary

namespace tt::test::ops::comparison
{

std::vector<tt::ops::Op> get_comparison_ops()
{
    return {
        tt::ops::OpType::Equal,
        tt::ops::OpType::Greater,
        tt::ops::OpType::GreaterEqual,
        tt::ops::OpType::Less,
        tt::ops::OpType::LessEqual,
        tt::ops::OpType::NotEqual,
    };
}

INSTANTIATE_TEST_SUITE_P(
    ComparisonOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_comparison_ops()),
            testing::ValuesIn(eltwise_binary::get_binary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    ComparisonOpsSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_comparison_ops()), testing::ValuesIn(eltwise_binary::generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::comparison

namespace tt::test::ops::logical_binary
{

std::vector<tt::ops::Op> get_logical_binary_ops()
{
    return {
        tt::ops::OpType::LogicalAnd,
    };
}

INSTANTIATE_TEST_SUITE_P(
    LogicalBinaryOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_logical_binary_ops()),
            testing::ValuesIn(eltwise_binary::get_binary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    LogicalBinaryOpsSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_logical_binary_ops()), testing::ValuesIn(eltwise_binary::generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::logical_binary

namespace tt::test::ops::logical_unary
{

std::vector<tt::ops::Op> get_logical_unary_ops()
{
    return {
        tt::ops::OpType::LogicalNot,
    };
}

INSTANTIATE_TEST_SUITE_P(
    LogicalUnaryOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_logical_unary_ops()),
            testing::ValuesIn(eltwise_unary::get_unary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    LogicalUnaryOpsSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_logical_unary_ops()), testing::ValuesIn(eltwise_unary::generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });
}  // namespace tt::test::ops::logical_unary

namespace tt::test::ops::bitwise_binary
{

std::vector<tt::ops::Op> get_bitwise_binary_ops()
{
    return {
        tt::ops::OpType::BitwiseAnd,
    };
}

INSTANTIATE_TEST_SUITE_P(
    BitwiseBinaryOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_bitwise_binary_ops()),
            testing::ValuesIn(eltwise_binary::get_binary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    BitwiseBinaryOpsSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_bitwise_binary_ops()), testing::ValuesIn(eltwise_binary::generate_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::bitwise_binary

namespace tt::test::ops::eltwise_nary
{

std::vector<tt::ops::Op> get_nary_eltwise_ops()
{
    return {
        tt::ops::OpType::Where,
    };
}

std::vector<VecShapes> get_nary_individual_test_shapes()
{
    return {
        // All the same shape.
        VecShapes{{1, 1, 1, 32}, {1, 1, 1, 32}, {1, 1, 1, 32}},
        VecShapes{{1, 1, 32, 1}, {1, 1, 32, 1}, {1, 1, 32, 1}},
        VecShapes{{1, 32, 1, 1}, {1, 32, 1, 1}, {1, 32, 1, 1}},
        VecShapes{{32, 1, 1, 1}, {32, 1, 1, 1}, {32, 1, 1, 1}},
        VecShapes{{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        VecShapes{{2, 3, 4}, {2, 3, 4}, {2, 3, 4}},
        VecShapes{{3, 4}, {3, 4}, {3, 4}},
        VecShapes{{4}, {4}, {4}},
        VecShapes{{1}, {1}, {1}},
        // Broadcasting cases.
        VecShapes{{1}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        VecShapes{{1, 1, 1, 1}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        VecShapes{{1, 1, 3, 1}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        // X or Y broadcasts.
        VecShapes{{1, 2, 3, 4}, {1}, {1, 2, 3, 4}},
        VecShapes{{1, 2, 3, 4}, {1, 2, 3, 4}, {1}},
    };
}

bool valid_nary_inputs(const graphlib::Shape& x, const graphlib::Shape& y, const graphlib::Shape& z)
{
    return graphlib::can_be_broadcasted(x, y) && graphlib::can_be_broadcasted(x, z) &&
           graphlib::can_be_broadcasted(y, z);
}

std::vector<std::vector<graphlib::Shape>> generate_nary_input_shapes()
{
    std::vector<std::vector<graphlib::Shape>> input_shapes;
    std::vector<graphlib::Shape> shapes;

    auto shapes_range = shape_range({1}, {8});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1}, {8, 8});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1}, {4, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());
    shapes_range = shape_range({1, 1, 1, 1}, {4, 1, 1, 1});
    shapes.insert(shapes.end(), shapes_range.begin(), shapes_range.end());

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        input_shapes.push_back({shapes[i], shapes[i], shapes[i]});

        for (size_t j = i + 1; j < shapes.size(); ++j)
        {
            for (size_t k = j; k < shapes.size(); ++k)
            {
                if (valid_nary_inputs(shapes[i], shapes[j], shapes[k]))
                {
                    input_shapes.push_back({shapes[i], shapes[j], shapes[k]});
                    if (i != j || j != k)
                    {
                        input_shapes.push_back({shapes[j], shapes[i], shapes[k]});
                        input_shapes.push_back({shapes[k], shapes[j], shapes[i]});
                    }
                }
            }
        }
    }

    return input_shapes;
}

INSTANTIATE_TEST_SUITE_P(
    NaryEltwiseOpsIndividual,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(
            testing::ValuesIn(get_nary_eltwise_ops()), testing::ValuesIn(get_nary_individual_test_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

INSTANTIATE_TEST_SUITE_P(
    NaryEltwiseOpsSweep,
    SimpleOpDecomposeOnlyTest,
    testing::ConvertGenerator(
        testing::Combine(testing::ValuesIn(get_nary_eltwise_ops()), testing::ValuesIn(generate_nary_input_shapes())),
        [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }),
    [](const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    { return SimpleOpDecomposeOnlyTest::get_test_name(info); });

}  // namespace tt::test::ops::eltwise_nary
