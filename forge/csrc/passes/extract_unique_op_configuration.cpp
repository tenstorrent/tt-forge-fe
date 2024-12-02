// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/extract_unique_op_configuration.hpp"

#include <filesystem>
#include <fstream>

#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

bool equivalent_shapes(OpShapesType vec1, OpShapesType vec2)
{
    if (vec1.size() != vec2.size())
        return false;
    for (size_t i = 0; i < vec1.size(); i++)
    {
        if (vec1[i] != vec2[i])
            return false;
    }
    return true;
}

UniqueOpShapesAttrsType extract_unique_op_configuration(
    graphlib::Graph *graph, const std::optional<std::vector<std::string>> &supported_ops)
{
    UniqueOpShapesAttrsType unique_op_shapes_attrs;
    OpShapesType operand_shapes;
    std::vector<std::string> supported_opnames;

    if (supported_ops.has_value())
        supported_opnames = supported_ops.value();

    for (graphlib::Node *current_node : graphlib::topological_sort(*graph))
    {
        auto current_op = dynamic_cast<graphlib::OpNode *>(current_node);
        if (not current_op)
        {
            continue;
        }

        graphlib::OpType current_node_optype = current_op->op_type();

        // If the list of op names (i.e supported_ops) is passed, it will only extract the
        // list of unique ops configuration that are not present in the supported_opnames,
        // otherwise it will extract all the unique op configurations in the graph
        if (!supported_opnames.empty())
        {
            if (std::find(supported_opnames.begin(), supported_opnames.end(), current_node_optype.op) !=
                supported_opnames.end())
                continue;
        }

        // Get the current node operand shapes
        operand_shapes.clear();
        for (auto operand : graph->data_operands(current_op))
        {
            operand_shapes.push_back(operand->shape());
        }

        // If the op is present in the unique_op_shapes_attrs map, then list of operand shapes
        // of the matched op is compared with the current node operand shapes otherwise the
        // current node operand_shapes and attributes(i.e OpTypes) are added to the unique_op_shapes_attrs map.
        if (unique_op_shapes_attrs.find(current_node_optype.op) != unique_op_shapes_attrs.end())
        {
            auto unique_shapes_attrs_list = unique_op_shapes_attrs.at(current_node_optype.op);
            bool operand_shapes_matched = false;
            for (size_t idx = 0; idx < unique_shapes_attrs_list.size(); idx++)
            {
                auto unique_shapes_attrs = unique_shapes_attrs_list.at(idx);
                auto unique_shapes = unique_shapes_attrs.first;
                auto unique_attrs = unique_shapes_attrs.second;

                // If the current node and matched op operand shapes are equivalent,
                // then take the list of OpType from the unique_shapes_attrs_list map with matched op name and
                // matched operand shapes and then compare with  current node optype, if the current
                // node optype is not present, then add current node optype in unique_op_shapes_attrs map with matched
                // op name and matched operand shapes.
                if (equivalent_shapes(unique_shapes, operand_shapes))
                {
                    operand_shapes_matched = true;
                    if (std::find(unique_attrs.begin(), unique_attrs.end(), current_node_optype) == unique_attrs.end())
                    {
                        unique_attrs.push_back(current_node_optype);
                        unique_op_shapes_attrs[current_node_optype.op].at(idx) = {unique_shapes, unique_attrs};
                        break;
                    }
                }
            }
            if (!operand_shapes_matched)
            {
                unique_op_shapes_attrs[current_node_optype.op].push_back({operand_shapes, {current_node_optype}});
            }
        }
        else
        {
            unique_op_shapes_attrs[current_node_optype.op] = {{operand_shapes, {current_node_optype}}};
        }
    }

    return unique_op_shapes_attrs;
}

void print_unique_op_configuration(const UniqueOpShapesAttrsType &unique_op_shapes_attrs, std::string op_config_info)
{
    std::cout << op_config_info << std::endl;
    for (auto unique_op_shapes_attrs_iter = unique_op_shapes_attrs.begin();
         unique_op_shapes_attrs_iter != unique_op_shapes_attrs.end();
         ++unique_op_shapes_attrs_iter)
    {
        auto op_name = unique_op_shapes_attrs_iter->first;
        auto shapes_attrs_list = unique_op_shapes_attrs_iter->second;
        std::cout << op_name << std::endl;
        for (auto shapes_attrs_iter = shapes_attrs_list.begin(); shapes_attrs_iter != shapes_attrs_list.end();
             ++shapes_attrs_iter)
        {
            auto op_shapes = shapes_attrs_iter->first;
            auto op_attrs = shapes_attrs_iter->second;
            std::cout << "\t\t Input_shape: [";
            for (size_t i = 0; i < op_shapes.size(); i++)
            {
                std::cout << op_shapes[i] << ", ";
            }
            std::cout << "]" << std::endl;
            if (!op_attrs.empty())
            {
                for (size_t i = 0; i < op_attrs.size(); i++)
                {
                    if (op_attrs[i].attr.size() > 0 or op_attrs[i].named_attrs.size() > 0)
                    {
                        std::cout << "\t\t\t\t\t Attributes: " << op_attrs[i].as_string() << std::endl;
                    }
                }
            }
        }
    }
}

std::string get_unique_op_configuration_file(std::string graph_name, std::string stage, std::string file_extension)
{
    std::string export_unique_op_config_default_dir_path = std::filesystem::current_path().string();
    std::string export_unique_op_config_dir_path =
        env_as<std::string>("FORGE_EXPORT_UNIQUE_OP_CONFIG_DIR_PATH", export_unique_op_config_default_dir_path);
    export_unique_op_config_dir_path = export_unique_op_config_dir_path + "/OpConfigs/" + graph_name + "/";
    if (not std::filesystem::exists(std::filesystem::path(export_unique_op_config_dir_path)))
    {
        TT_ASSERT(
            std::filesystem::create_directories(std::filesystem::path(export_unique_op_config_dir_path)),
            "Export Directory creation failed!");
    }
    std::string export_unique_op_config_file = export_unique_op_config_dir_path + stage + file_extension;
    return export_unique_op_config_file;
}

void export_unique_op_configuration_to_csv_file(
    const UniqueOpShapesAttrsType &unique_op_shapes_attrs, std::string graph_name, std::string stage)
{
    std::string export_unique_op_config_file = get_unique_op_configuration_file(graph_name, stage, ".csv");
    std::string delimiter = env_as<std::string>("FORGE_EXPORT_UNIQUE_OP_CONFIG_CSV_DELIMITER", "/");
    std::string headers = "OpName" + delimiter + "Shape" + delimiter + "Attributes";
    log_info(
        "Exporting unique ops configuration in {} compilation stage to {} file", stage, export_unique_op_config_file);

    std::fstream fs;
    fs.open(export_unique_op_config_file, std::fstream::out | std::fstream::trunc);
    fs << headers << "\n";

    for (auto unique_op_shapes_attrs_iter = unique_op_shapes_attrs.begin();
         unique_op_shapes_attrs_iter != unique_op_shapes_attrs.end();
         ++unique_op_shapes_attrs_iter)
    {
        auto op_name = unique_op_shapes_attrs_iter->first;
        auto shapes_attrs_list = unique_op_shapes_attrs_iter->second;
        for (auto shapes_attrs_iter = shapes_attrs_list.begin(); shapes_attrs_iter != shapes_attrs_list.end();
             ++shapes_attrs_iter)
        {
            auto op_shapes = shapes_attrs_iter->first;
            auto op_attrs = shapes_attrs_iter->second;
            if (!op_attrs.empty())
            {
                for (size_t i = 0; i < op_attrs.size(); i++)
                {
                    fs << op_name << delimiter << "[";
                    for (size_t j = 0; j < op_shapes.size(); j++)
                    {
                        fs << op_shapes[j] << ", ";
                    }
                    fs << "]" << delimiter;
                    if (op_attrs[i].attr.size() > 0 or op_attrs[i].named_attrs.size() > 0)
                    {
                        fs << op_attrs[i].as_string();
                    }
                    fs << "\n";
                }
            }
            else
            {
                fs << op_name << delimiter << "[";
                for (size_t i = 0; i < op_shapes.size(); i++)
                {
                    fs << op_shapes[i] << ", ";
                }
                fs << "]" << delimiter;
                fs << "\n";
            }
        }
    }

    fs.close();
}

void export_unique_op_configuration_to_xlsx_file(
    const UniqueOpShapesAttrsType &unique_op_shapes_attrs, std::string graph_name, std::string stage)
{
    std::string export_unique_op_config_file = get_unique_op_configuration_file(graph_name, stage, ".xlsx");

    py::module_ forge_utils_module = py::module_::import("forge.utils");

    py::str py_export_unique_op_config_file_path = py::str(export_unique_op_config_file);
    py::str py_graph_name = py::str(graph_name);
    py::str py_stage = py::str(stage);
    py::dict py_unique_op_shapes_attrs;

    using UniqueOpShapesAttrsPairType = std::pair<std::string, std::vector<OpShapesAttrsType>>;
    for (UniqueOpShapesAttrsPairType opnames_and_shapes_attrs : unique_op_shapes_attrs)
    {
        std::string op_name = opnames_and_shapes_attrs.first;
        std::vector<OpShapesAttrsType> unique_shapes_attrs = opnames_and_shapes_attrs.second;

        py::list py_shapes_attrs;
        for (OpShapesAttrsType shapes_attrs : unique_shapes_attrs)
        {
            OpShapesType shapes = shapes_attrs.first;
            OpAttrsType attrs = shapes_attrs.second;

            py::list py_shapes;
            for (graphlib::Shape shape : shapes)
            {
                py_shapes.append(shape.as_vector());
            }
            py::list py_attrs;
            if (!attrs.empty())
            {
                for (graphlib::OpType attr : attrs)
                {
                    if (attr.attr.size() > 0 or attr.named_attrs.size() > 0)
                        py_attrs.append(attr.as_string());
                }
            }
            py_shapes_attrs.append(py::make_tuple(py_shapes, py_attrs));
        }
        py_unique_op_shapes_attrs[py::str(op_name)] = py_shapes_attrs;
    }

    // Call the export_unique_op_configuration_to_xslx_file Python function
    py::bool_ export_status = forge_utils_module.attr("create_xlsx_file_from_unique_op_config")(
        py_unique_op_shapes_attrs, py_graph_name, py_stage, py_export_unique_op_config_file_path);

    if (export_status)
    {
        log_info(
            "Successfully exported unique ops configuration in {} compilation stage to {} xlsx file",
            stage,
            export_unique_op_config_file);
    }
    else
    {
        log_warning(
            "Problem in exporting unique ops configuration in {} compilation stage to {} xlsx file",
            stage,
            export_unique_op_config_file);
    }
}

void extract_unique_op_configuration(
    graphlib::Graph *graph, std::string stage, const std::optional<std::vector<std::string>> &supported_ops)
{
    auto stage_to_extract = env_as_optional<std::string>("FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT");
    bool print_unique_op_config = env_as<bool>("FORGE_PRINT_UNIQUE_OP_CONFIG", false);
    std::string export_unique_op_config_file_type = env_as<std::string>("FORGE_EXPORT_UNIQUE_OP_CONFIG_FILE_TYPE", "");
    std::transform(
        export_unique_op_config_file_type.begin(),
        export_unique_op_config_file_type.end(),
        export_unique_op_config_file_type.begin(),
        ::tolower);

    if (not stage_to_extract or ((stage_to_extract != stage) and (stage_to_extract != "ALL")))
        return;

    UniqueOpShapesAttrsType unique_op_shapes_attrs = extract_unique_op_configuration(graph, supported_ops);

    if (unique_op_shapes_attrs.empty())
        return;

    if (print_unique_op_config)
    {
        std::string op_config_info = std::string("Op Configuration at: ") + stage;
        print_unique_op_configuration(unique_op_shapes_attrs, op_config_info);
    }

    if (export_unique_op_config_file_type == "csv")
    {
        export_unique_op_configuration_to_csv_file(unique_op_shapes_attrs, graph->name(), stage);
    }
    else if (export_unique_op_config_file_type == "xlsx")
    {
        export_unique_op_configuration_to_xlsx_file(unique_op_shapes_attrs, graph->name(), stage);
    }
}

}  // namespace tt::passes
