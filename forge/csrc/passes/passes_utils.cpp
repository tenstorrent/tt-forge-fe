// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes_utils.hpp"

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt
{

using NodeType = graphlib::NodeType;

bool divisible_either_direction(int a, int b) { return (a % b == 0) or (b % a == 0); }

void optimize_tms(std::vector<graphlib::OpType> &tms)
{
    if (tms.size() < 2)
    {
        return;
    }

    enum class Erase
    {
        None,
        B,
        AB,
    };

    using graphlib::OpType;
    using MatchFn = std::function<bool(OpType const &, OpType const &)>;  // return true means apply MergeFn
    using MergeFn = std::function<Erase(OpType &, OpType &)>;
    std::vector<std::pair<MatchFn, MergeFn>> rules = {
        // hoist transpose before broadcast
        {[](OpType const &a, OpType const &b) { return a.op == "broadcast" and b.op == "transpose"; },
         [](OpType &a, OpType &b)
         {
             int &dim = std::get<int>(a.attr[0]);
             if (dim > 1)
             {
                 TT_ASSERT(dim == 2 or dim == 3);
                 dim = (dim == 2) ? 3 : 2;
             }
             std::swap(a, b);
             return Erase::None;
         }},

        // back to back broadcast (order c, r, z)
        {[](OpType const &a, OpType const &b) {
             return (a.op == "broadcast" and b.op == "broadcast") and
                    (std::get<int>(a.attr[0]) < std::get<int>(b.attr[0]));
         },
         [](OpType &a, OpType &b)
         {
             std::swap(a, b);
             return Erase::None;
         }},

        // back to back transpose
        {[](OpType const &a, OpType const &b) { return a.op == "transpose" and b.op == "transpose"; },
         [](OpType &, OpType &) { return Erase::AB; }},

        // back to back select
        {[](OpType const &a, OpType const &b) { return false and a.op == "select" and b.op == "select"; },
         [](OpType &a, OpType &b)
         {
             TT_ASSERT(a.attr.size() == 4 and b.attr.size() == 4);
             // TODO: there are some cases of back to back select that can be merged, when a.length >= b.length
             return Erase::None;
         }},
    };

    bool any_updated = true;
    while (any_updated)
    {
        any_updated = false;
        for (auto [match_fn, merge_fn] : rules)
        {
            auto iter = std::adjacent_find(tms.begin(), tms.end(), match_fn);
            if (iter != tms.end())
            {
                any_updated = true;
                switch (merge_fn(*iter, *(iter + 1)))
                {
                    case Erase::B:
                    {
                        tms.erase(iter + 1);
                        break;
                    }
                    case Erase::AB:
                    {
                        tms.erase(iter, iter + 2);
                        break;
                    }
                    default: break;
                }
            }
        }
    }
}

void optimize_tms(Graph *graph)
{
    for (Node *node : graph->nodes())
    {
        if (node->node_type() == NodeType::kForgeOp)
        {
            for (auto const &edge : graph->operand_data_edges(node))
            {
                auto edge_attributes = graph->get_edge_attributes(edge);
                std::vector<graphlib::OpType> &tms = edge_attributes->get_tms();
                // Collapse mergeable tms
                optimize_tms(tms);
            }
        }
    }
}

// Recalculate all node shapes from inputs
void recalculate_shapes(graphlib::Graph *graph)
{
    for (Node *n : graphlib::topological_sort(*graph))
    {
        if (n->node_type() == graphlib::NodeType::kInput)
            continue;

        graphlib::calculate_and_set_node_shape(graph, n);
    }
}

std::vector<int> get_factors(int num)
{
    std::vector<int> factors;

    while (num % 2 == 0)
    {
        factors.push_back(2);
        num /= 2;
    }

    int sqrt_num = sqrt(num);
    for (int i = 3; i <= sqrt_num; i += 2)
    {
        while (num % i == 0)
        {
            factors.push_back(i);
            num /= i;
        }
    }

    if (num > 2)
    {
        factors.push_back(num);
    }

    return factors;
}

bool check_unsupported_hw_ops(Graph *graph, bool should_throw)
{
    bool unsupported_hw_ops = false;
    std::string message;

    for (Node *node : graph->nodes())
    {
        // TODO: Remove this block once backend supports hconcat / vconcat
        if (node->node_type() == NodeType::kForgeNaryTM)
        {
            graphlib::ForgeNaryTMNode *tm = node->as<graphlib::ForgeNaryTMNode>();
            unsupported_hw_ops = true;
            message += fmt::format("{} {}\n", tm->name(), tm->op_type().op);
            continue;
        }

        if (node->node_type() != NodeType::kForgeOp)
            continue;
    }

    if (unsupported_hw_ops and should_throw)
        throw UnsupportedHWOpsError(message);

    return unsupported_hw_ops;
}

// Returns true if string is part of 2D vector of strings, false otherwise.
bool is_str_in_strings(const std::string &str, const std::vector<std::vector<std::string>> &strings)
{
    for (const std::vector<std::string> &iter : strings)
    {
        if (std::find(iter.begin(), iter.end(), str) != iter.end())
        {
            return true;
        }
    }

    return false;
}

}  // namespace tt
