// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "autograd/binding.hpp"

#include <vector>

std::tuple<Shape, std::vector<DimBroadcast>> get_op_shape(OpType type, std::vector<Shape> &operands)
{
    std::vector<std::vector<std::uint32_t>> operand_tuples;
    for (Shape &shape : operands) operand_tuples.push_back(shape.as_vector());

    return type.new_op().shape(operand_tuples);
}

NodeContext insert_backward(
    autograd_context context,
    OpType type,
    int operand,
    const std::vector<NodeContext> &inputs,
    NodeContext output,
    NodeContext gradient)
{
    auto eval_module = py::module_::import("forge.op.eval.forge");
    py::function forge_backward = eval_module.attr("get_f_forge_backward")(type);

    return forge_backward(context, operand, inputs, output, gradient).cast<NodeContext>();
}
