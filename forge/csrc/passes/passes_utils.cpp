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
