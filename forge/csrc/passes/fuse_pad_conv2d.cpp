// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/commute_utils.hpp"
#include "passes/consteval.hpp"
#include "python_bindings_common.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

// Find consecutive Pad and Conv2D nodes and fuse them into a single Conv2D node.
void fuse_pad_conv2d(graphlib::Graph *graph)
{
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op or op->op_name() != "pad")
        {
            continue;
        }

        auto attrs = op->named_attrs();
        auto padding_variant = attrs["padding"];

        if (std::get<std::string>(attrs["mode"]) != "constant")
        {
            continue;
        }

        // Check if padding is already a vector of ints
        std::vector<int> *padding_vec = std::get_if<std::vector<int>>(&padding_variant);
        if (!padding_vec)
        {
            // If not, make it a vector and populate with default padding values
            padding_vec = new std::vector<int>(std::get<int>(attrs["pad_len"]), 0);
            attrs["padding"] = *padding_vec;  // Update the "padding" in the attributes map
        }

        // Extract padding values, assuming the padding vector has at least 4 values

        int top_pad = (*padding_vec)[-4];     // Top padding (index -4)
        int bottom_pad = (*padding_vec)[-3];  // Bottom padding (index -3)
        int left_pad = (*padding_vec)[-2];    // Left padding (index -2)
        int right_pad = (*padding_vec)[-1];   // Right padding (index -1)

        auto users = graph->users(node);
        bool all_users_are_conv2d = true;

        for (auto *user : users)
        {
            graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(user);
            if (not user_op or user_op->op_name() != "conv2d")
            {
                all_users_are_conv2d = false;
                break;
            }
        }

        if (not all_users_are_conv2d)
            continue;

        // Add Pad to Conv2d attributes
        for (auto user : users)
        {
            graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(user);
            graphlib::OpType::Attrs conv_attrs = user_op->named_attrs();
            TT_ASSERT(conv_attrs.size() == 7 && "Expected 7 attributes in conv2d op but got {}", conv_attrs.size());

            // Conv2d attributes [stride, dilation, groups, padding]
            auto &conv_padding_variant = conv_attrs["padding"];

            // Ensure that the padding is a vector
            std::vector<int> *conv_padding_vec = std::get_if<std::vector<int>>(&conv_padding_variant);
            if (!conv_padding_vec)
            {
                // If not a vector, we need to convert it to a vector
                conv_padding_vec = new std::vector<int>(4, 0);  // Assuming 4 padding values (top, left, bottom, right)
                conv_attrs["padding"] = *conv_padding_vec;      // Update the padding attribute
            }
            // Adding up the pad values with the existing conv2d pad va;ues
            (*conv_padding_vec)[0] += top_pad;     // Top
            (*conv_padding_vec)[1] += left_pad;    // Left
            (*conv_padding_vec)[2] += bottom_pad;  // Bottom
            (*conv_padding_vec)[3] += right_pad;   // Right

            std::vector<int> int_conv_attrs;
            for (const auto &attr : conv_attrs)
            {
                // Extract only integer attributes
                if (std::holds_alternative<int>(attr.second))
                {
                    int_conv_attrs.push_back(std::get<int>(attr.second));
                }
            }

            update_conv_attr(user_op, int_conv_attrs);
        }

        // Bypass the pad node
        graphlib::bypass_node(graph, node, true);
    }
}
}  // namespace tt::passes
