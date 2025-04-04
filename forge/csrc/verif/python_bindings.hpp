// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pybind11/pybind11.h"
#pragma clang diagnostic pop
namespace py = pybind11;

namespace tt
{

void VerifModule(py::module &m_verif);

}  // namespace tt
