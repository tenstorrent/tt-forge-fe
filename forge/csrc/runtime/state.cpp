// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "runtime/state.hpp"

#include <utils/logger.hpp>
#include <vector>

#include "runtime/runtime.hpp"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

namespace tt
{

std::ostream& operator<<(std::ostream& os, ProgramType program_type)
{
    switch (program_type)
    {
        case ProgramType::Forward: os << "Forward"; break;
        case ProgramType::Backward: os << "Backward"; break;
        case ProgramType::Optimizer: os << "Optimizer"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

ProgramState create_program_state(
    ProgramType program_type, const TensorPool& tensor_pool, std::vector<std::string> persistent_input_names)
{
    std::vector<tt::Tensor> persistent_inputs;
    persistent_inputs.reserve(persistent_input_names.size());

    for (auto& name : persistent_input_names)
    {
        auto tensor = tensor_pool.get_tensor(name);
        persistent_inputs.emplace_back(tensor);
    }
    return ProgramState{program_type, persistent_inputs, {}};
}

void ModelState::run_program(ProgramType program_type, std::vector<tt::Tensor> act_inputs)
{
    // ISSUE(#1346): So far, the device_id is hardcoded to 0 - make it user-configurable.
    constexpr size_t device_id = 0;
    size_t pg_id = program_idx(program_type);
    std::optional<ProgramState>& opt_program_state = program_states[pg_id];

    TT_ASSERT(opt_program_state.has_value(), "Program state for {} not initialized", program_type);

    if (!TTSystem::get_system().devices[device_id]->is_open())
    {
        TTSystem::get_system().devices[device_id]->open_device();
    }

    auto& program_state = opt_program_state.value();

    // Clear the outputs from the previous run.
    program_state.outputs.clear();

    std::vector<tt::Tensor> inputs;
    inputs.reserve(act_inputs.size() + program_state.persistent_inputs.size());

    // Push activation inputs to the device if they are not already there.
    // NOTE: there is an ordering requirement for the activation inputs and the persistent inputs, i.e.
    // in the input list (`inputs` vector here), the activation inputs come first. Unfortunately, there isn't any
    // mechanism which enforces this, yet. It's an informal contract between the compiler and the runtime.
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

    // Push persistent inputs to the device if they are not already there.
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

bool ModelState::test_so(
    std::string so_path,
    std::string func_name,
    std::vector<tt::Tensor>& act_inputs,
    std::vector<tt::Tensor>& consts_and_params,
    std::vector<tt::Tensor>& golden_outs)
{
    void* so_handle = tt::open_so(so_path);
    std::vector<tt::runtime::Tensor> outs = tt::run_so_program(so_handle, func_name, act_inputs, consts_and_params);

    std::vector<runtime::Tensor> host_outs;
    std::transform(
        outs.begin(),
        outs.end(),
        std::back_inserter(host_outs),
        [](tt::runtime::Tensor& output)
        {
            std::vector<runtime::Tensor> vec_host_t = tt::runtime::toHost(output);
            assert(vec_host_t.size() == 1 && "We expect only one host tensor");
            return vec_host_t.front();
        });

    std::vector<runtime::Tensor> golden_host_outs;
    golden_host_outs.reserve(golden_outs.size());

    std::transform(
        golden_outs.begin(),
        golden_outs.end(),
        std::back_inserter(golden_host_outs),
        [](tt::Tensor& output)
        {
            runtime::Tensor rt_t = output.get_runtime_tensor();
            std::vector<runtime::Tensor> vec_host_t = tt::runtime::toHost(rt_t);
            assert(vec_host_t.size() == 1 && "We expect only one host tensor");
            return vec_host_t.front();
        });

    tt::close_so(so_handle);

    return tt::compareOuts(host_outs, golden_host_outs);
}
};  // namespace tt
