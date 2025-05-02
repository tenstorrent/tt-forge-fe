// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/pybind11.h>
#pragma clang diagnostic pop

#include <optional>
#include <ostream>
#include <string>

namespace py = pybind11;

namespace tt::property
{

enum class ExecutionDepth
{
    CI_FAILURE,                 // CI failure (e.g. download of model weights fails)
    FAILED_FE_COMPILATION,      // Front end compilation fails (tvm->ttir)
    FAILED_TTMLIR_COMPILATION,  // TT-MLIR compilation fails, can't produce flatbuffer
    FAILED_RUNTIME,             // Runtime execution fails
    INCORRECT_RESULT,           // Flatbuffer executed, output incorrect
    PASSED,                     // Outputs correct (pcc >= 0.99)
};

std::string to_string(const ExecutionDepth depth);

std::ostream& operator<<(std::ostream& os, const ExecutionDepth depth);

void record_execution_depth(
    const ExecutionDepth execution_depth, const std::optional<py::object>& forge_property_handler);

void record_flatbuffer_details(
    const std::string& binary_json_str, const std::optional<py::object>& forge_property_handler);

}  // namespace tt::property
