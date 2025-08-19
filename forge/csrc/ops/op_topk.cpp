// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <tuple>
#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "ops/op_common.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace topk
{
using namespace graphlib;

// Attributes expected:
//  - k: int (required)
//  - dim: int (required)
//  - largest: bool (optional; default true)
//  - sorted: bool (optional; default true)

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::TopK, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "TopK should have one input tensor");

    const int64_t k = static_cast<int64_t>(op.attr_as<int>("k"));
    const int64_t dim = static_cast<int64_t>(op.attr_as<int>("dim"));
    const bool largest = op.has_attr("largest") ? op.attr_as<bool>("largest") : true;
    const bool sorted = op.has_attr("sorted") ? op.attr_as<bool>("sorted") : true;

    // torch::topk returns a tuple (values, indices). Our infra is single-output; return values for now.
    auto result = torch::topk(tensors[0], k, dim, largest, sorted);
    at::Tensor values = std::get<0>(result);
    // at::Tensor indices = std::get<1>(result); // kept for future multi-output support

    return values;
}

std::vector<at::Tensor> eval_multi(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::TopK, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "TopK should have one input tensor");

    const int64_t k = static_cast<int64_t>(op.attr_as<int>("k"));
    const int64_t dim = static_cast<int64_t>(op.attr_as<int>("dim"));
    const bool largest = op.has_attr("largest") ? op.attr_as<bool>("largest") : true;
    const bool sorted = op.has_attr("sorted") ? op.attr_as<bool>("sorted") : true;

    auto result = torch::topk(tensors[0], k, dim, largest, sorted);
    at::Tensor values = std::get<0>(result);
    at::Tensor indices = std::get<1>(result);
    return {values, indices};
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::TopK, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "TopK should have one input shape");

    const auto &input = in_shapes[0];
    TT_ASSERT(!input.empty(), "TopK input must have rank >= 1");

    const int dim = op.attr_as<int>("dim");
    TT_ASSERT(dim >= -static_cast<int>(input.size()) && dim < static_cast<int>(input.size()), "TopK dim out of range");

    const int pos_dim = dim < 0 ? dim + static_cast<int>(input.size()) : dim;
    std::vector<uint32_t> out_shape = input;
    out_shape[pos_dim] = static_cast<uint32_t>(op.attr_as<int>("k"));

    return {Shape::create(out_shape), {}};
}

std::tuple<std::vector<Shape>, std::vector<DimBroadcast>> shape_multi(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    auto [single, bcast] = shape(old_op_type, op, in_shapes);
    return {std::vector<Shape>{single, single}, bcast};
}

// No autograd for now

tt::graphlib::NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::TopK, "Wrong op type.");
    TT_THROW(false, "TopK does not have backward.");
    unreachable();
}

}  // namespace topk
}  // namespace ops
}  // namespace tt
