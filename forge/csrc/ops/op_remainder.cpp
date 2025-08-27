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
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace remainder
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_ASSERT(tensors.size() == 2, "Remainder should have two input tensors.");
    return torch::remainder(tensors[0], tensors[1]);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Remainder, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 2, "Remainder should have two input shapes.");
    TT_ASSERT(op.attrs().size() == 0, "Remainder should not have any attrs.");

    return op_common::compute_elementwise_binary_shape(in_shapes);
}

NodeContext backward(

    const Op &op,
    tt::autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_ASSERT(false, "Remainder does not have backward.");
    unreachable();
}

}  // namespace remainder
}  // namespace ops
}  // namespace tt
