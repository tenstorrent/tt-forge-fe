// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// Interface to python definitions of op shapes & backward generators

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"

using namespace tt::autograd;
using Shape = tt::graphlib::Shape;
using OpType = tt::graphlib::OpType;
using DimBroadcast = tt::graphlib::DimBroadcast;
using TileDim = tt::TileDim;

std::tuple<Shape, std::vector<DimBroadcast>> get_op_shape(
    OpType type, std::vector<Shape> &operands, TileDim tile_dim = TileDim::Dim32x32);

NodeContext insert_backward(
    autograd_context context,
    OpType type,
    int operand,
    const std::vector<NodeContext> &inputs,
    NodeContext output,
    NodeContext gradient);
