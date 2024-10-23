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

void export_unique_op_configuration_to_csv_file(
    const UniqueOpShapesAttrsType &unique_op_shapes_attrs, std::string graph_name, std::string stage)
{
    std::string export_unique_op_config_default_path = std::filesystem::current_path().string();
    std::string export_unique_op_config_path =
        env_as<std::string>("FORGE_EXPORT_UNIQUE_OP_CONFIG_DIR_PATH", export_unique_op_config_default_path);
    export_unique_op_config_path = export_unique_op_config_path + "/OpConfigs/" + graph_name + "/";
    if (not std::filesystem::exists(std::filesystem::path(export_unique_op_config_path)))
    {
        TT_ASSERT(
            std::filesystem::create_directories(std::filesystem::path(export_unique_op_config_path)),
            "Export Directory creation failed!");
    }
    std::string export_unique_op_config_file = export_unique_op_config_path + stage + ".csv";
    std::string headers = "OpName-Operands Shape-Attributes";
    std::string delimiter = "-";

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

void extract_unique_op_configuration(
    graphlib::Graph *graph, std::string stage, const std::optional<std::vector<std::string>> &supported_ops)
{
    auto stage_to_extract = env_as_optional<std::string>("FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT");
    bool print_unique_op_config = env_as<bool>("FORGE_PRINT_UNIQUE_OP_CONFIG", false);
    bool export_unique_op_config_to_csv = env_as<bool>("FORGE_EXPORT_UNIQUE_OP_CONFIG_TO_CSV", false);

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
    if (export_unique_op_config_to_csv)
    {
        export_unique_op_configuration_to_csv_file(unique_op_shapes_attrs, graph->name(), stage);
    }
}

}  // namespace tt::passes
