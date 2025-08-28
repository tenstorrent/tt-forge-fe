// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "autograd/binding.hpp"

#include <vector>

std::tuple<Shape, std::vector<DimBroadcast>> get_op_shape(Op op, std::vector<Shape> &operands)
{
    std::vector<std::vector<std::uint32_t>> operand_tuples;
    for (Shape &shape : operands) operand_tuples.push_back(shape.as_vector());

    return op.shape(operand_tuples);
}
