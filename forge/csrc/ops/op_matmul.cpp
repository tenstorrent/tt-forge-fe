// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace matmul
{

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "Matmul should have 2 input tensors.");

    // Get the tensors and cast them for CPU evaluation
    auto t0 = tensors[0];
    auto t1 = tensors[1];

    // Perform matrix multiplication
    at::Tensor result = torch::matmul(t0, t1);

    return result;
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "Matmul should have 2 input tensors.");

    std::vector<std::uint32_t> shape0 = in_shapes[0];
    std::vector<std::uint32_t> shape1 = in_shapes[1];

    // Handle special cases for 1D tensors following torch.matmul rules
    if (shape0.size() == 1 && shape1.size() == 1)
    {
        // Both 1D: dot product -> scalar (empty shape)
        return std::make_tuple(graphlib::Shape::create({}), std::vector<graphlib::DimBroadcast>());
    }

    bool shape0_was_1d = shape0.size() == 1;
    bool shape1_was_1d = shape1.size() == 1;

    // Prepend 1 to 1D tensors to enable matrix multiplication
    if (shape0_was_1d)
    {
        shape0.insert(shape0.begin(), 1);
    }
    if (shape1_was_1d)
    {
        shape1.push_back(1);
    }

    // Now both tensors are at least 2D
    TT_ASSERT(shape0.size() >= 2 && shape1.size() >= 2, "Tensors should be at least 2D after preprocessing");

    // Check matrix multiplication compatibility
    TT_ASSERT(shape0.back() == shape1[shape1.size() - 2], "Matrix dimensions must be compatible for multiplication");

    // Pad shapes to same length for batch dimension broadcasting
    int in0_padding = 0;
    while (shape0.size() < shape1.size())
    {
        shape0.insert(shape0.begin(), 1);
        in0_padding++;
    }

    int in1_padding = 0;
    while (shape1.size() < shape0.size())
    {
        shape1.insert(shape1.begin(), 1);
        in1_padding++;
    }

    std::vector<graphlib::DimBroadcast> broadcast;
    std::vector<std::uint32_t> output_shape;

    // Handle batch dimension broadcasting (all dimensions except the last 2)
    for (int i = 0; i < static_cast<int>(shape0.size()) - 2; i++)
    {
        std::uint32_t dim0 = shape0[i];
        std::uint32_t dim1 = shape1[i];

        if (dim0 == dim1)
        {
            // Same size, no broadcast needed
            output_shape.push_back(dim0);
        }
        else if (dim0 == 1)
        {
            // Broadcast operand 0
            broadcast.push_back(std::make_tuple(0, i - in0_padding, dim1));
            output_shape.push_back(dim1);
        }
        else if (dim1 == 1)
        {
            // Broadcast operand 1
            broadcast.push_back(std::make_tuple(1, i - in1_padding, dim0));
            output_shape.push_back(dim0);
        }
        else
        {
            // Incompatible dimensions
            TT_ASSERT(false, "Batch dimensions must be broadcastable");
        }
    }

    // Matrix dimensions: output is [batch_dims...], shape0[-2], shape1[-1]
    output_shape.push_back(shape0[shape0.size() - 2]);
    output_shape.push_back(shape1[shape1.size() - 1]);

    // Remove dimensions that were added for 1D tensor handling
    if (shape0_was_1d)
    {
        // Remove the prepended dimension from output
        output_shape.erase(output_shape.end() - 2);
    }
    if (shape1_was_1d)
    {
        // Remove the appended dimension from output
        output_shape.pop_back();
    }

    return std::make_tuple(graphlib::Shape::create(output_shape), broadcast);
}

tt::graphlib::NodeContext backward(
    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient)
{
    TT_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "Matmul should have two inputs for backward pass.");
    TT_ASSERT(operand < 2, "Invalid operand index");

    tt::graphlib::NodeContext in0 = inputs[0];
    tt::graphlib::NodeContext in1 = inputs[1];

    if (operand == 0)
    {
        // d/dA (A @ B) = grad @ B.T
        graphlib::OpType transpose_op("transpose");
        transpose_op.set_attr("dim0", -2);
        transpose_op.set_attr("dim1", -1);

        tt::graphlib::NodeContext in1_t = ac.autograd->create_op(ac, transpose_op, {in1});

        graphlib::OpType matmul_op("matmul");
        return ac.autograd->create_op(ac, matmul_op, {gradient, in1_t});
    }
    else
    {
        // d/dB (A @ B) = A.T @ grad
        graphlib::OpType transpose_op("transpose");
        transpose_op.set_attr("dim0", -2);
        transpose_op.set_attr("dim1", -1);

        tt::graphlib::NodeContext in0_t = ac.autograd->create_op(ac, transpose_op, {in0});

        graphlib::OpType matmul_op("matmul");
        return ac.autograd->create_op(ac, matmul_op, {in0_t, gradient});
    }
}

void decompose_initial(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    // No initial decomposition needed for matmul
}

void decompose_post_optimize(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    // No post-optimize decomposition needed for matmul
}

void decompose_post_autograd(const Op &op, DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs)
{
    // No post-autograd decomposition needed for matmul
}

long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");

    auto [output_shape, broadcast] = shape(op, in_shapes);
    std::vector<std::uint32_t> out_shape = output_shape.as_vector();

    long macc = 0;
    if (out_shape.size() >= 2)
    {
        macc = out_shape[out_shape.size() - 1] * out_shape[out_shape.size() - 2];
        if (out_shape.size() > 2)
        {
            macc *= out_shape[out_shape.size() - 3];
        }
        if (out_shape.size() > 3)
        {
            macc *= out_shape[out_shape.size() - 4];
        }
        macc *= in_shapes[0][in_shapes[0].size() - 1];
    }

    return macc * 2;  // 2 FLOPs per MAC operation
}

}  // namespace matmul
}  // namespace ops
}  // namespace tt
