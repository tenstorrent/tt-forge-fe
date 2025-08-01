// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
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
namespace argmax
{
using namespace graphlib;

at::Tensor eval(const graphlib::OpType &old_op_type, const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Argmax, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Argmax should have one input");

    std::optional<int64_t> dim = std::nullopt;
    if (op.has_attr("dim_arg"))
        dim = op.attr_as<std::vector<int>>("dim_arg")[0];

    return torch::argmax(tensors[0], dim, op.attr_as<bool>("keep_dim"));
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const graphlib::OpType &old_op_type, const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Argmax, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Argmax should have one input");

    if (op.has_attr("dim_arg"))
        return op_common::reduce_ops_shape(op, in_shapes);

    if (op.attr_as<bool>("keep_dim"))
        return {Shape::create(std::vector<uint32_t>(in_shapes[0].size(), 1U)), {}};

    return {Shape::create({}), {}};
}

NodeContext backward(
    const graphlib::OpType &old_op_type,
    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Argmax, "Wrong op type.");
    TT_ASSERT(false, "Argmax does not have backward.");
    unreachable();
}

}  // namespace argmax
}  // namespace ops
}  // namespace tt
