// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op.hpp"
#include "torch/extension.h"
#include "torch/torch.h"

namespace tt
{
namespace ops
{

torch::Tensor OpAdd::eval(const std::vector<torch::Tensor> &tensors) const
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

Shape OpAdd::shape(const std::vector<std::vector<std::uint32_t>> &in_shapes) const
{
    TT_ASSERT(in_shapes.size() == 1, "OpAbs::shape should have single input shape.");
    return Shape::create(in_shapes[0]);
}

long OpAdd::initial_flops_stimate(const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    Shape out_shape = shape(inputs);
    return std::accumulate(out_shape.begin(), out_shape.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace ops
}  // namespace tt
