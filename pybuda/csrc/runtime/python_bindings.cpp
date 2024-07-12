// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime/python_bindings.hpp"
#include "runtime/runtime.hpp"
#include "tt/runtime/types.h"

namespace tt {

void RuntimeModule(py::module &m_runtime)
{
    py::class_<runtime::Binary>(m_runtime, "Binary")
        .def("get_program_inputs", &runtime::Binary::getProgramInputs)
        .def("get_program_outputs", &runtime::Binary::getProgramOutputs);
    m_runtime.def("run_binary", tt::run_binary);
}

} // namespace tt
