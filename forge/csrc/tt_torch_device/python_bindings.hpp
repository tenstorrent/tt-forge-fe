// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/pybind11.h>
#pragma clang diagnostic pop

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace tt {

void TorchDeviceModule(py::module &m_torch_device);

}  // namespace tt
