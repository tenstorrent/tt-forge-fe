// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime/python_bindings.hpp"

#include "runtime/runtime.hpp"
#include "runtime/state.hpp"
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

    py::class_<Tensor>(m_runtime, "Tensor").def(py::init<torch::Tensor&>()).def("to_torch", &Tensor::to_host);
    py::class_<TensorPool>(m_runtime, "TensorPool")
        .def(py::init<>())
        .def("get_tensor", &TensorPool::get_tensor)
        .def(
            "insert",
            [](TensorPool& self, const std::string& name, torch::Tensor& tensor) { self.insert(name, tensor); })
        .def("update_tensor", &TensorPool::update_tensor);

    py::enum_<ProgramType>(m_runtime, "ProgramType")
        .value("Forward", ProgramType::Forward)
        .value("Backward", ProgramType::Backward)
        .value("Optimizer", ProgramType::Optimizer)
        .export_values();

    py::class_<ProgramState>(m_runtime, "ProgramState")
        .def(py::init<ProgramType, std::vector<Tensor>, std::vector<Tensor>>());

    py::class_<ModelState>(m_runtime, "ModelState")
        .def(py::init<runtime::Binary>())
        .def_property_readonly("tensor_pool", &ModelState::get_tensor_pool)
        .def(
            "init_program_state",
            [](ModelState& self, ProgramState& program_state) { self.init_program_state(program_state); })
        .def(
            "run_program",
            [](ModelState& self, ProgramType program_type, std::vector<tt::Tensor>& act_inputs)
            { self.run_program(program_type, act_inputs); })
        .def(
            "get_outputs",
            [](ModelState& self, ProgramType program_type)
            { return self.program_states[program_idx(program_type)].outputs; });

    m_runtime.def("create_program_state", &tt::create_program_state);
}

}  // namespace tt
