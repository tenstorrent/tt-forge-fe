// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "tt/runtime/types.h"

namespace tt
{

enum class ProgramType : uint32_t
{
    Forward = 0,
    Backward,
    Optimizer,
    Count  // Keep this last
};

constexpr size_t PROGRAM_TYPE_COUNT = static_cast<std::underlying_type_t<ProgramType>>(ProgramType::Count);

constexpr auto program_idx(ProgramType program_type)
{
    TT_ASSERT(program_type < ProgramType::Count, "Invalid program type");
    return static_cast<std::underlying_type_t<ProgramType>>(program_type);
}

struct ProgramState
{
    ProgramType program_type;
    std::vector<tt::Tensor> persistent_inputs;
    std::vector<tt::Tensor> outputs;
};

ProgramState create_program_state(ProgramType program_type, std::vector<tt::Tensor> persistent_inputs);

struct ModelState
{
    runtime::Binary binary;
    std::array<ProgramState, PROGRAM_TYPE_COUNT> program_states;
    TensorPool tensor_pool;

    TensorPool& get_tensor_pool() { return tensor_pool; }

    ModelState(runtime::Binary binary) : binary{binary}, program_states{}, tensor_pool{} {}

    ModelState(const ModelState&) = delete;

    void init_program_state(ProgramState& program_state)
    {
        auto program_type = program_state.program_type;
        auto pidx = program_idx(program_type);

        program_states[pidx] = program_state;
    }

    void run_program(ProgramType program_type, std::vector<tt::Tensor> act_inputs);
};

}  // namespace tt
