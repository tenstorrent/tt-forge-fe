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
    TT_ASSERT(tensors.size() >= 2 && tensors.size() <= 3, "Matmul should have 2 or 3 input tensors.");

    bool accumulate = op.has_attr("accumulate") ? op.attr_as<bool>("accumulate") : false;

    // Get the tensors and cast them for CPU evaluation
    auto t0 = tensors[0];
    auto t1 = tensors[1];

    // Handle different data types for CPU evaluation
    at::ScalarType original_type = t0.scalar_type();
    if (t0.scalar_type() == at::ScalarType::BFloat16)
    {
        t0 = t0.to(at::ScalarType::Float);
        t1 = t1.to(at::ScalarType::Float);
    }

    // Perform matrix multiplication
    at::Tensor result = torch::matmul(t0, t1);

    // Convert back to original type
    result = result.to(original_type);

    // Add bias if present
    if (tensors.size() > 2)
    {
        result = result + tensors[2];
    }

    // Apply accumulation if requested
    if (accumulate && result.dim() >= 3)
    {
        result = torch::sum(result, -3, true);
    }

    return result;
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");
    TT_ASSERT(in_shapes.size() >= 2 && in_shapes.size() <= 4, "Matmul should have 2 to 4 input shapes.");

    bool accumulate = op.has_attr("accumulate") ? op.attr_as<bool>("accumulate") : false;

    std::vector<std::uint32_t> shape0 = in_shapes[0];
    std::vector<std::uint32_t> shape1 = in_shapes[1];

    int ops0_padding = 0;
    while (shape0.size() < shape1.size())
    {
        shape0.insert(shape0.begin(), 1);
        ops0_padding++;
    }

    int ops1_padding = 0;
    while (shape1.size() < shape0.size())
    {
        shape1.insert(shape1.begin(), 1);
        ops1_padding++;
    }

    std::vector<graphlib::DimBroadcast> broadcast;
    std::vector<std::uint32_t> output_shape;

    // Handle higher dimensions (beyond 3rd)
    for (int dim = 4; dim <= static_cast<int>(shape0.size()); dim++)
    {
        int idx = shape0.size() - dim;
        TT_ASSERT(shape0[idx] == shape1[idx], "Broadcast on dimensions beyond 3rd is not supported");
        output_shape.insert(output_shape.begin(), shape0[idx]);
    }

    // Handle Z dimension broadcast
    if (shape0.size() >= 3)
    {
        int z_idx = shape0.size() - 3;
        if (shape0[z_idx] != shape1[z_idx])
        {
            if (shape0[z_idx] == 1)
            {
                broadcast.push_back(std::make_tuple(0, static_cast<int>(shape0.size()) - 3, shape1[z_idx]));
                output_shape.insert(output_shape.begin(), shape1[z_idx]);
            }
            else if (shape1[z_idx] == 1)
            {
                broadcast.push_back(std::make_tuple(1, static_cast<int>(shape0.size()) - 3, shape0[z_idx]));
                output_shape.insert(output_shape.begin(), shape0[z_idx]);
            }
            else
            {
                TT_ASSERT(false, "If Z dimension is not the same for matmul, one of operands must have it be 1.");
            }
        }
        else
        {
            output_shape.insert(output_shape.begin(), shape0[z_idx]);
        }
    }

    // Handle inner dimension broadcast
    int last_idx = shape0.size() - 1;
    int second_last_idx = shape0.size() - 2;

    if (shape0[last_idx] != shape1[second_last_idx])
    {
        if (shape0[last_idx] == 1)
        {
            broadcast.push_back(std::make_tuple(0, last_idx - ops0_padding, shape1[second_last_idx]));
        }
        else if (shape1[second_last_idx] == 1)
        {
            broadcast.push_back(std::make_tuple(1, second_last_idx - ops1_padding, shape0[last_idx]));
        }
        else
        {
            TT_ASSERT(false, "If inner dimension is not the same for matmul, one of operands must have it be 1");
        }
    }

    // Output dimensions: [batch_dims...], shape0[-2], shape1[-1]
    output_shape.push_back(shape0[second_last_idx]);
    output_shape.push_back(shape1[last_idx]);

    if (accumulate)
    {
        TT_ASSERT(output_shape.size() >= 3, "Accumulate requires at least 3 dimensions");
        output_shape[output_shape.size() - 3] = 1;
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
