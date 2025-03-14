// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "forge_property_utils.hpp"

#include "utils/assert.hpp"

namespace tt::property
{

// Convert ExecutionDepth enum to string
std::string to_string(const ExecutionDepth depth)
{
    switch (depth)
    {
        case ExecutionDepth::FAILED_FE_COMPILATION: return "FAILED_FE_COMPILATION";
        case ExecutionDepth::FAILED_TTMLIR_COMPILATION: return "FAILED_TTMLIR_COMPILATION";
        case ExecutionDepth::FAILED_RUNTIME: return "FAILED_RUNTIME";
        case ExecutionDepth::INCORRECT_RESULT: return "INCORRECT_RESULT";
        case ExecutionDepth::PASSED: return "PASSED";
        default: TT_ASSERT(false, "Invalid ExecutionDepth");
    }
    return "";
}

std::ostream& operator<<(std::ostream& os, const ExecutionDepth depth) { return os << to_string(depth); }

void record_execution_depth(
    const std::optional<py::object>& forge_property_handler, const ExecutionDepth execution_depth)
{
    // If the optional handler is not provided or explicitly set to None in Python, exit early.
    if (!forge_property_handler.has_value() || forge_property_handler->is_none())
    {
        return;
    }

    // Ensure that the provided handler object has a 'record_execution_depth' method.
    if (!py::hasattr(*forge_property_handler, "record_execution_depth"))
    {
        throw std::runtime_error("The provided forge_property_handler does not have a record_execution_depth method");
    }

    // Invoke the 'record_execution_depth' method of the Python object, passing the execution depth.
    forge_property_handler->attr("record_execution_depth")(execution_depth);
}

}  // namespace tt::property
