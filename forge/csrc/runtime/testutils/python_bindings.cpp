// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime/testutils/python_bindings.hpp"

#include "runtime/state.hpp"
#include "runtime/tensor.hpp"
#include "runtime/testutils/runtime_testutils.hpp"

namespace tt
{

void RuntimeTestUtilsModule(py::module &m_runtime_testutils)
{
    m_runtime_testutils
        .def(
            "test_so",
            [](std::string so_path,
               std::string func_name,
               std::vector<tt::Tensor> &inputs,
               std::vector<tt::Tensor> &consts_and_params,
               std::vector<tt::Tensor> &outputs)
            { return tt::runtime_testutils::test_so(so_path, func_name, inputs, consts_and_params, outputs); })
        .def(
            "get_persistent_inputs",
            [](ProgramType program_type, ModelState &model_state) -> std::vector<tt::Tensor>
            {
                std::optional<ProgramState> &opt_program_state = model_state.program_states[program_idx(program_type)];
                TT_ASSERT(opt_program_state.has_value(), "Program state for {} not initialized", program_type);
                return opt_program_state.value().persistent_inputs;
            });
}

}  // namespace tt
