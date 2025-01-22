// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime/python_bindings.hpp"

#include "runtime/runtime.hpp"
#include "tt/runtime/types.h"

namespace tt
{

void RuntimeModule(py::module& m_runtime)
{
    py::class_<runtime::Binary>(m_runtime, "Binary")
        .def("get_program_inputs", &runtime::Binary::getProgramInputs)
        .def("get_program_outputs", &runtime::Binary::getProgramOutputs);
    m_runtime.def("run_binary", tt::run_binary);
    m_runtime.def("run_binary_v2", tt::run_binary_v2);

    py::class_<Tensor>(m_runtime, "Tensor").def(py::init<torch::Tensor&>()).def(py::init<runtime::Tensor&>());
    py::class_<TensorPool>(m_runtime, "TensorPool")
        .def(py::init<>())
        .def("get_tensor", &TensorPool::get_tensor)
        .def(
            "insert",
            [](TensorPool& self, const std::string& name, torch::Tensor& tensor) { self.insert(name, tensor); })
        .def("update_tensor", &TensorPool::update_tensor);
}

}  // namespace tt
