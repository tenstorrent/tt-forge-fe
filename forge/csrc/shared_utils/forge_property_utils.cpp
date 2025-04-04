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

// Helper function: converts a std::vector<std::uint32_t> to a py::tuple.
py::tuple vector_to_pytuple(const std::vector<std::uint32_t>& vec)
{
    py::tuple tuple_obj(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        tuple_obj[i] = py::cast(vec[i]);
    }
    return tuple_obj;
}

void record_singleop_operands_info(
    const std::optional<py::object>& forge_property_handler, const tt::graphlib::Graph* graph)
{
    // Retrieve nodes of type kPyOp.
    std::vector<tt::graphlib::Node*> ops = graph->nodes_by_type(graphlib::NodeType::kPyOp);
    if (ops.size() != 1)
    {
        throw std::runtime_error("Expected one operation(i.e NodeType: kPyOp) inside the forge graph");
    }

    std::vector<OperandType> operands_info;
    for (auto operand : graph->data_operands(ops[0]))
    {
        tt::graphlib::InputNode* input_node = operand->as<graphlib::InputNode>();
        std::string input_type = input_node->input_type_string();
        input_type = (input_type == "input") ? "activation" : input_type;
        operands_info.emplace_back(input_type, input_node->shape().as_vector(), input_node->output_df());
    }

    py::list py_operands_info;
    for (const auto& [input_type, shape_vec, data_format] : operands_info)
    {
        py_operands_info.append(py::make_tuple(py::str(input_type), vector_to_pytuple(shape_vec), data_format));
    }

    // Validate the handler; exit early if it fails.
    if (!validate_handler_method(forge_property_handler, "record_operands_info"))
    {
        return;
    }

    forge_property_handler->attr("record_operands_info")(py_operands_info);
}

}  // namespace tt::property
