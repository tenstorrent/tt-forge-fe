// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_common.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace matmul
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");
    TT_ASSERT(tensors.size() == 2, "Matmul should have two inputs.");

    std::vector<at::Tensor> promoted_tensors = op_common::promote_floating_dtypes(tensors);
    at::ScalarType t0_type = promoted_tensors[0].scalar_type();
    at::ScalarType t1_type = promoted_tensors[1].scalar_type();

    // Matmuls of int8 results in int32.
    at::ScalarType result_type =
        t0_type == torch::kI8 || t1_type == torch::kI8 ? torch::kI32 : at::promote_types(t0_type, t1_type);

    // Promotes tensors to float32.
    auto promote = [](const at::Tensor &t)
    {
        if (t.scalar_type() == torch::kF16 || t.scalar_type() == torch::kI8)
            return t.to(torch::kF32);

        return t;
    };

    auto t0 = promote(promoted_tensors[0]);
    auto t1 = promote(promoted_tensors[1]);

    return torch::matmul(t0, t1).to(result_type);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(const Op &op, const std::vector<std::vector<uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "Matmul should have two inputs.");

    auto input0_shape = in_shapes[0];
    auto input1_shape = in_shapes[1];
    int dim0 = static_cast<int>(input0_shape.size());
    int dim1 = static_cast<int>(input1_shape.size());

    std::vector<uint32_t> output_shape;

    /**
     * 1D x 1D: dot product -> scalar.
     */
    if (dim0 == 1 && dim1 == 1)
        return {Shape::create(output_shape), {}};

    /**
     * 1D x nD: vector-matrix multiplication.
     * Prepend 1 to input0, then remove first dim from result.
     * [K] x [..., K, N] -> [..., N]
     */
    if (dim0 == 1 && dim1 >= 2)
    {
        for (int i = 0; i < dim1 - 2; ++i) output_shape.push_back(input1_shape[i]);

        output_shape.push_back(input1_shape[dim1 - 1]);  // N dimension
        return {Shape::create(output_shape), {}};
    }

    /**
     * nD x 1D: matrix-vector multiplication
     * [..., M, K] x [K] -> [..., M]
     */
    if (dim0 >= 2 && dim1 == 1)
    {
        for (int i = 0; i < dim0 - 2; ++i) output_shape.push_back(input0_shape[i]);

        output_shape.push_back(input0_shape[dim0 - 2]);  // M dimension
        return {Shape::create(output_shape), {}};
    }

    /**
     * Standard nD x nD matrix multiplication
     * [..., M, K] x [..., K, N] -> [..., M, N]
     */

    int max_batch_dims = std::max(dim0, dim1) - 2;

    // Broadcast batch dimensions from left to right (excluding last 2 dims)
    for (int i = 0; i < max_batch_dims; ++i)
    {
        int idx0 = i - (max_batch_dims - dim0 + 2);
        int idx1 = i - (max_batch_dims - dim1 + 2);

        uint32_t batch_dim0 = (idx0 >= 0) ? input0_shape[idx0] : 1;
        uint32_t batch_dim1 = (idx1 >= 0) ? input1_shape[idx1] : 1;

        // Standard broadcasting: dimensions must be equal or one of them must be 1
        uint32_t out_dim = std::max(batch_dim0, batch_dim1);
        output_shape.push_back(out_dim);
    }

    // Add matrix dimensions: M from input0, N from input1
    output_shape.push_back(input0_shape[dim0 - 2]);  // M (rows from first input)
    output_shape.push_back(input1_shape[dim1 - 1]);  // N (cols from second input)

    return {Shape::create(output_shape), {}};
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Matmul, "Wrong op type.");
    TT_ASSERT(inputs.size() == 2, "Matmul should have two inputs for backward pass.");
    TT_ASSERT(operand < 2, "Invalid operand index");

    if (operand == 0)
    {
        tt::graphlib::NodeContext in1_t =
            ac.autograd->create_op(ac, Op("transpose", {{"dim0", -2}, {"dim1", -1}}), {inputs[1]});

        return ac.autograd->create_op(ac, Op("matmul"), {gradient, in1_t});
    }

    tt::graphlib::NodeContext in0_t =
        ac.autograd->create_op(ac, Op("transpose", {{"dim0", -2}, {"dim1", -1}}), {inputs[0]});

    return ac.autograd->create_op(ac, Op("matmul"), {in0_t, gradient});
}

}  // namespace matmul
}  // namespace ops
}  // namespace tt
