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
        case ExecutionDepth::CI_FAILURE: return "CI_FAILURE";
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

// Helper function to validate that the handler is provided and contains the required method.
// Returns true if the handler is valid; false otherwise.
bool validate_handler_method(const std::optional<py::object>& handler, const std::string& method_name)
{
    // If the optional handler is not provided or explicitly set to None in Python, return false
    if (!handler.has_value() || handler->is_none())
    {
        return false;
    }
    // Ensure that the provided handler object has a 'method_name' method.
    if (!py::hasattr(*handler, method_name.c_str()))
    {
        return false;
    }
    return true;
}

void record_execution_depth(
    const std::optional<py::object>& forge_property_handler, const ExecutionDepth execution_depth)
{
    // Validate the handler; exit early if it fails.
    if (!validate_handler_method(forge_property_handler, "record_execution_depth"))
    {
        return;
    }

    // Invoke the 'record_execution_depth' method of the Python object, passing the execution depth.
    forge_property_handler->attr("record_execution_depth")(execution_depth);
}

void record_flatbuffer_details(
    const std::optional<py::object>& forge_property_handler, const std::string& binary_json_str)
{
    // Validate the handler; exit early if it fails.
    if (!validate_handler_method(forge_property_handler, "record_flatbuffer_details"))
    {
        return;
    }

    // Invoke the 'record_flatbuffer_details' method of the Python object, passing the binary json string.
    forge_property_handler->attr("record_flatbuffer_details")(binary_json_str);
}

}  // namespace tt::property
