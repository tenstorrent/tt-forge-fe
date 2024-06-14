// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "autograd/binding.hpp"

#include <vector>

std::tuple<Shape, std::vector<DimBroadcast>> get_op_shape(OpType type, std::vector<Shape> &operands, bool is_buda, TileDim tile_dim)
{
    int tile_height = tt::graphlib::get_row_size_from_tile_size(tile_dim);
    int tile_width = tt::graphlib::get_col_size_from_tile_size(tile_dim);
    auto eval_module = is_buda ? py::module_::import("pybuda.op.eval.buda") : py::module_::import("pybuda.op.eval.pybuda");
    py::function pybuda_shape = is_buda ? eval_module.attr("get_f_pybuda_shape")(type, tile_height, tile_width)
                                        : eval_module.attr("get_f_pybuda_shape")(type);

    std::vector<std::vector<std::uint32_t>> operand_tuples;
    for(Shape &shape : operands)
        operand_tuples.push_back(shape.as_vector());

    py::tuple ret = pybuda_shape(operand_tuples);
    Shape s = is_buda ? Shape::create_buda(ret[0].cast<std::vector<std::uint32_t>>(), tile_height, tile_width) : 
                        Shape::create(ret[0].cast<std::vector<std::uint32_t>>());

    return std::make_tuple(s, ret[1].cast<std::vector<DimBroadcast>>());
}

NodeContext insert_backward(
    autograd_context context,
    OpType type,
    int operand,
    const std::vector<NodeContext> &inputs,
    NodeContext output,
    NodeContext gradient)
{
    auto eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function pybuda_backward = eval_module.attr("get_f_pybuda_backward")(type);

    return pybuda_backward(context, operand, inputs, output, gradient).cast<NodeContext>();
}

