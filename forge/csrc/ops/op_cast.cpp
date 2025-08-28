// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_forge/common.hpp"
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
namespace cast
{
using namespace graphlib;

at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors)
{
    TT_DBG_ASSERT(op.type() == OpType::Cast, "Wrong op type.");
    TT_ASSERT(tensors.size() == 1, "Cast should have one input.");
    TT_ASSERT(op.attrs().size() == 1, "Cast should have 1 attribute: dtype.");

    DataFormat df = static_cast<DataFormat>(op.attr_as<int>("dtype"));
    at::ScalarType target_type = data_format_to_scalar_type(df);
    return tensors[0].to(target_type);
}

std::tuple<Shape, std::vector<DimBroadcast>> shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    TT_DBG_ASSERT(op.type() == OpType::Cast, "Wrong op type.");
    TT_ASSERT(in_shapes.size() == 1, "Cast should have one input.");
    TT_ASSERT(op.attrs().size() == 1, "Cast should have 1 attribute: dtype.");

    return {Shape::create(in_shapes[0]), {}};
}

NodeContext backward(

    const Op &op,
    autograd::autograd_context &ac,
    int operand,
    const std::vector<NodeContext> &inputs,
    const NodeContext &output,
    const NodeContext &gradient)
{
    TT_DBG_ASSERT(op.type() == OpType::Cast, "Wrong op type.");
    TT_ASSERT(inputs.size() == 1, "Cast should have one input.");
    TT_ASSERT(operand == 0, "Invalid operand index for cast.");
    TT_ASSERT(op.attrs().size() == 1, "Cast should have 1 attribute: dtype.");

    TT_ASSERT(false, "Cast does not have backward.");
    unreachable();
}

}  // namespace cast
}  // namespace ops
}  // namespace tt
