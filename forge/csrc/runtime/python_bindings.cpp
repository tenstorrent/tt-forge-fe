// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime/python_bindings.hpp"

#include "runtime/runtime.hpp"
#include "runtime/tt_device.hpp"
#include "tt/runtime/types.h"

namespace tt
{

void RuntimeModule(py::module &m_runtime)
{
    // Main runtime APIs
    py::class_<runtime::Binary>(m_runtime, "Binary")
        .def("get_program_inputs", &runtime::Binary::getProgramInputs)
        .def("get_program_outputs", &runtime::Binary::getProgramOutputs);
    m_runtime.def("run_binary", tt::run_binary);

    // Experimental APIs
    py::module m_experimental = m_runtime.def_submodule("experimental");
    py::class_<tt::DeviceSettings>(m_experimental, "DeviceSettings")
        .def(py::init<>())
        .def_readwrite("enable_program_cache", &tt::DeviceSettings::enable_program_cache);
    m_experimental.def(
        "configure_devices",
        [](DeviceSettings settings) -> void { tt::TTSystem::get_system().configure_devices(settings); },
        py::arg("device_settings") = DeviceSettings(),
        "Configure all devices with the given settings.");
}

}  // namespace tt
