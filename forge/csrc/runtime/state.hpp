// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <optional>

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

std::ostream& operator<<(std::ostream& os, ProgramType program_type);

constexpr size_t PROGRAM_TYPE_COUNT = static_cast<std::underlying_type_t<ProgramType>>(ProgramType::Count);

constexpr auto program_idx(ProgramType program_type)
{
    TT_ASSERT(program_type < ProgramType::Count, "Invalid program type");
    return static_cast<std::underlying_type_t<ProgramType>>(program_type);
}

// Simple struct to hold the state of a program.
struct ProgramState
{
    ProgramType program_type;
    std::vector<tt::Tensor> persistent_inputs;
    std::vector<tt::Tensor> outputs;
};

// Encapsulates execution context for a model.
struct ModelState
{
    // Handle to the flatbuffer binary which contains the model.
    runtime::Binary binary;

    // Static array of program states, one for each program type.
    std::array<std::optional<ProgramState>, PROGRAM_TYPE_COUNT> program_states;

    // Tensor pool containing tensors shared between different programs or program invocations.
    // E.g. weights, constants, etc.
    //
    // These tensors will be pushed to the device the first time they are used in a program (as inputs to the program).
    // They will be kept on the device until the runtime/user decides to remove them from the device. In that case on
    // next use they will be pushed to the device again.
    TensorPool tensor_pool;

    ModelState(runtime::Binary binary) : binary{binary}, program_states{}, tensor_pool{} {}

    // Disallow copy construction and assignment.
    ModelState(const ModelState&) = delete;
    ModelState(ModelState&&) = default;
    ModelState& operator=(const ModelState&) = delete;

    TensorPool& get_tensor_pool() { return tensor_pool; }

    void add_program_state(ProgramState& program_state)
    {
        auto program_type = program_state.program_type;
        auto pidx = program_idx(program_type);

        program_states[pidx] = program_state;
    }

    void run_program(ProgramType program_type, std::vector<tt::Tensor> act_inputs);

    bool test_so(
        std::string so_path,
        std::string func_name,
        std::vector<tt::Tensor>& act_inputs,
        std::vector<tt::Tensor>& consts_and_params,
        std::vector<tt::Tensor>& outputs);
};

ProgramState create_program_state(
    ProgramType program_type, const TensorPool& tensor_pool, std::vector<std::string> persistent_input_names);

}  // namespace tt
