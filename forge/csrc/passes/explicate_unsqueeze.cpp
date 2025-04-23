// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/explicate_unsqueeze.hpp"

#include <pybind11/pybind11.h>

#include <functional>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"
#include "passes/print_graph.hpp"

namespace tt::passes
{

void hoist_unsqueeze_squeeze_to_reshape(graphlib::Graph *graph)
{
    std::unordered_set<graphlib::Node *> nodes_to_remove;
    for (auto *node : graphlib::topological_sort(*graph))
    {
        auto op = dynamic_cast<graphlib::OpNode *>(node);

        if (not op)
        {
            continue;
        }
        if (op->op_name() != "reshape")
        {
            continue;
        }
        // Find reshape -> unsqueeze pattern and replace with reshape
        auto users = graph->users(node);
        auto operands = graph->operands(node);
        if (users.size() != 1 or operands.size() != 1)
        {
            continue;
        }

        auto user_op = dynamic_cast<graphlib::OpNode *>(users[0]);
        auto operand_op = dynamic_cast<graphlib::OpNode *>(operands[0]);
        bool user_is_squeeze_unsqueeze =
            (user_op and (user_op->op_name() == "unsqueeze" or user_op->op_name() == "squeeze"));
        bool operand_is_squeeze_unsqueeze =
            (operand_op and (operand_op->op_name() == "unsqueeze" or operand_op->op_name() == "squeeze"));
        if (not user_is_squeeze_unsqueeze and not operand_is_squeeze_unsqueeze)
        {
            continue;
        }

        if (user_is_squeeze_unsqueeze)
        {
            auto target_shape = user_op->shape().as_vector();
            std::vector<graphlib::OpType::Attr> new_reshape_attr;
            for (auto dim : target_shape)
            {
                new_reshape_attr.push_back((int)dim);
            }
            std::vector<int> shape_vector(target_shape.begin(), target_shape.end());
            graphlib::OpType::Attrs named_attrs;
            named_attrs["shape"] = shape_vector;
            op->change_op_type(graphlib::OpType("reshape", new_reshape_attr, {}, named_attrs));
            op->set_shape(user_op->shape());
            nodes_to_remove.insert(users[0]);
        }

        if (operand_is_squeeze_unsqueeze)
        {
            nodes_to_remove.insert(operands[0]);
        }
    }

    for (auto node : nodes_to_remove)
    {
        auto maintain_tms = [graph](graphlib::Edge new_edge, graphlib::Edge original_edge)
        {
            auto original_tms = graph->get_edge_attributes(original_edge)->get_tms();
            auto new_tms = graph->get_edge_attributes(new_edge)->get_tms();
            new_tms.insert(new_tms.end(), original_tms.begin(), original_tms.end());
        };
        bypass_node(graph, node, true, maintain_tms);
    }
    recalculate_shapes(graph);
}

}  // namespace tt::passes
