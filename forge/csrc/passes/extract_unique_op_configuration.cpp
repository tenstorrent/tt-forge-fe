// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/extract_unique_op_configuration.hpp"

#include <filesystem>
#include <fstream>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

bool equivalent_shapes(std::vector<graphlib::Shape> vec1, std::vector<graphlib::Shape> vec2)
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

void extract_unique_op_configuration(
    graphlib::Graph *graph, std::string stage, const std::optional<std::vector<std::string>> &supported_ops)
{
    auto stage_to_extract = env_as_optional<std::string>("PYBUDA_EXTRACT_UNIQUE_OP_CONFIG_AT");
    bool print_unique_op_config = env_as<bool>("PYBUDA_PRINT_UNIQUE_OP_CONFIG", false);
    auto export_unique_op_config_to_csv = env_as<bool>("PYBUDA_EXPORT_UNIQUE_OP_CONFIG_TO_CSV", false);

    if (not stage_to_extract or ((stage_to_extract != stage) and (stage_to_extract != "ALL")))
        return;

    std::unordered_map<std::string, std::unordered_map<std::uint64_t, std::vector<graphlib::Shape>>> unique_op_shapes;
    std::unordered_map<std::string, std::unordered_map<std::uint64_t, std::vector<graphlib::OpType>>> unique_op_attrs;
    std::vector<graphlib::Shape> operand_shapes;
    std::vector<std::string> supported_opnames;
    std::uint64_t unique_id = 0;

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

        // If the list of op names (i.e supported_ops) is passed, it will only print the
        // list of unique ops configuration that are not present in the supported_opnames,
        // otherwise it will print all the unique op configurations in the graph
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

        // If the op is present in the unique_op_shapes and unique_op_attrs unordered map,
        // then list of operand shapes of the matched op is compared with the current node operand shapes
        // otherwise the current node operand_shapes and attributes(i.e OpTypes) are added to the
        // unique_op_shapes and unique_op_attrs with new unique_id.
        if ((unique_op_shapes.find(current_node_optype.op) != unique_op_shapes.end()) &&
            (unique_op_attrs.find(current_node_optype.op) != unique_op_attrs.end()))
        {
            auto unique_op_id_and_opshapes = unique_op_shapes.at(current_node_optype.op);
            auto unique_op_id_and_optypes = unique_op_attrs.at(current_node_optype.op);
            bool operand_shapes_matched = false;
            for (auto iter = unique_op_id_and_opshapes.begin(); iter != unique_op_id_and_opshapes.end(); ++iter)
            {
                auto unique_op_id = iter->first;
                auto unique_op_shapes = iter->second;
                // If the current node and matched op operand shapes are equivalent,
                // then take the list of OpType from the unique_op_attrs unordered map with matched op name and
                // unique_id from the matched operand shapes and then compare with  current node optype, if the current
                // node optype is not present, then add current node optype in unique_op_attrs with matched op name and
                // unique_id from matched operand shapes.
                if (equivalent_shapes(unique_op_shapes, operand_shapes) &&
                    (unique_op_id_and_optypes.find(unique_op_id) != unique_op_id_and_optypes.end()))
                {
                    operand_shapes_matched = true;
                    auto unique_optypes = unique_op_id_and_optypes.at(unique_op_id);
                    if (std::find(unique_optypes.begin(), unique_optypes.end(), current_node_optype) ==
                        unique_optypes.end())
                    {
                        unique_op_attrs[current_node_optype.op][unique_op_id].push_back(current_node_optype);
                    }
                }
            }
            if (!operand_shapes_matched)
            {
                unique_op_shapes[current_node_optype.op][unique_id] = operand_shapes;
                unique_op_attrs[current_node_optype.op][unique_id] = {current_node_optype};
                unique_id++;
            }
        }
        else
        {
            unique_op_shapes[current_node_optype.op] = {{unique_id, operand_shapes}};
            unique_op_attrs[current_node_optype.op] = {{unique_id, {current_node_optype}}};
            unique_id++;
        }
    }

    if (unique_op_shapes.empty() and unique_op_attrs.empty())
        return;

    // print the unique ops configurations that are present in the gragh
    if (print_unique_op_config)
    {
        std::cout << "Op Configuration at: " << stage << std::endl;
        for (auto iter1 = unique_op_shapes.begin(); iter1 != unique_op_shapes.end(); ++iter1)
        {
            auto op_name = iter1->first;
            std::cout << op_name << std::endl;
            for (auto iter2 = iter1->second.begin(); iter2 != iter1->second.end(); ++iter2)
            {
                auto op_id = iter2->first;
                auto op_shapes = iter2->second;
                std::cout << "\t\t Input_shape: [";
                for (size_t i = 0; i < op_shapes.size(); i++)
                {
                    std::cout << op_shapes[i] << ", ";
                }
                std::cout << "]" << std::endl;
                auto has_op_and_id = (unique_op_attrs.find(op_name) != unique_op_attrs.end()) &&
                                     (unique_op_attrs[op_name].find(op_id) != unique_op_attrs[op_name].end());
                if (has_op_and_id)
                {
                    auto optypes = unique_op_attrs[op_name][op_id];
                    for (size_t i = 0; i < optypes.size(); i++)
                    {
                        if (optypes[i].attr.size() > 0 or optypes[i].named_attrs.size() > 0)
                        {
                            std::cout << "\t\t\t\t\t Attributes: " << optypes[i].as_string() << std::endl;
                        }
                    }
                }
            }
        }
    }

    // Export the unique ops configurations to CSV file(delimiter: - )
    if (export_unique_op_config_to_csv)
    {
        std::string export_unique_op_config_default_path = std::filesystem::current_path().string();
        std::string export_unique_op_config_path =
            env_as<std::string>("PYBUDA_EXPORT_UNIQUE_OP_CONFIG_DIR_PATH", export_unique_op_config_default_path);
        export_unique_op_config_path = export_unique_op_config_path + "/OpConfigs/" + graph->name() + "/";
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
            "Exporting unique ops configuration in {} compilation stage to {} file",
            stage,
            export_unique_op_config_file);

        std::fstream fs;
        fs.open(export_unique_op_config_file, std::fstream::out | std::fstream::trunc);
        fs << headers << "\n";

        for (auto iter1 = unique_op_shapes.begin(); iter1 != unique_op_shapes.end(); ++iter1)
        {
            auto op_name = iter1->first;
            for (auto iter2 = iter1->second.begin(); iter2 != iter1->second.end(); ++iter2)
            {
                auto op_id = iter2->first;
                auto op_shapes = iter2->second;
                auto has_op_and_id = (unique_op_attrs.find(op_name) != unique_op_attrs.end()) &&
                                     (unique_op_attrs[op_name].find(op_id) != unique_op_attrs[op_name].end());
                if (has_op_and_id)
                {
                    auto optypes = unique_op_attrs[op_name][op_id];
                    for (size_t i = 0; i < optypes.size(); i++)
                    {
                        fs << op_name << delimiter << "[";
                        for (size_t j = 0; j < op_shapes.size(); j++)
                        {
                            fs << op_shapes[j] << ", ";
                        }
                        fs << "]" << delimiter;
                        if (optypes[i].attr.size() > 0 or optypes[i].named_attrs.size() > 0)
                        {
                            fs << optypes[i].as_string();
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
}

}  // namespace tt::passes
