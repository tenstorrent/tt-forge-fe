// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "runtime/state.hpp"

#include <utils/logger.hpp>

#include "tt/runtime/runtime.h"

namespace tt
{

ProgramState create_program_state(ProgramType program_type, std::vector<tt::Tensor> persistent_inputs)
{
    return ProgramState{program_type, persistent_inputs, {}};
}

void ModelState::run_program(ProgramType program_type, std::vector<tt::Tensor> act_inputs)
{
    constexpr size_t device_id = 0;
    size_t pg_id = program_idx(program_type);
    auto& program_state = program_states[pg_id];

    if (!TTSystem::get_system().devices[device_id]->is_open())
    {
        TTSystem::get_system().devices[device_id]->open_device();
    }

    for (auto& tensor : program_state.outputs)
    {
        runtime::deallocateTensor(tensor.get_runtime_tensor(), true);
    }

    program_state.outputs.clear();

    std::vector<tt::Tensor> inputs;
    inputs.reserve(act_inputs.size() + program_state.persistent_inputs.size());

    size_t input_idx = 0;
    for (auto tensor : act_inputs)
    {
        if (!tensor.on_device())
        {
            auto layout = tt::runtime::getLayout(binary, pg_id, input_idx++);
            tensor.to_device(device_id, layout);
        }

        inputs.emplace_back(tensor);
    }

    for (auto& persistent_input : program_state.persistent_inputs)
    {
        size_t curr_input_id = input_idx++;
        if (!persistent_input.on_device())
        {
            auto layout = tt::runtime::getLayout(binary, pg_id, curr_input_id);
            persistent_input.to_device(device_id, layout);
        }

        inputs.emplace_back(persistent_input);
    }

    program_state.outputs = ::tt::run_program(binary, pg_id, inputs);
}
};  // namespace tt
