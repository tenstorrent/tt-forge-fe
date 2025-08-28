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
using Op = tt::ops::Op;
using DimBroadcast = tt::graphlib::DimBroadcast;
using TileDim = tt::TileDim;

std::tuple<Shape, std::vector<DimBroadcast>> get_op_shape(Op op, std::vector<Shape> &operands);
