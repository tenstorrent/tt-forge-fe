// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tuple>

#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "torch/extension.h"
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{

at::Tensor Op::add_eval(const std::vector<at::Tensor> &tensors) const
{
    // assert len(tensors) == 1, "Abs should have one input"
    // shape = tensors[0].shape
    // original_types = [o.dtype for o in tensors]
    // ret = torch.abs(tensors[0])

    // if ret.dtype != original_types[0]:
    // ret = ret.type(original_types[0])

    // return ret

    TT_ASSERT(tensors.size() == 1, "OpAbs::eval should have single input tensor.");
    return torch::abs(tensors[0]);
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::add_shape(
    const std::vector<std::vector<std::uint32_t>> &in_shapes) const
{
    TT_ASSERT(in_shapes.size() == 1, "OpAbs::shape should have single input shape.");
    return std::make_tuple(graphlib::Shape::create(in_shapes[0]), std::vector<graphlib::DimBroadcast>{});
}

long Op::add_initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    auto shape_tuple = shape(inputs);
    graphlib::Shape out_shape = std::get<0>(shape_tuple);

    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace ops
}  // namespace tt
