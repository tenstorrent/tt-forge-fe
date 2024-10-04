// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

#include "utils/assert.hpp"

namespace tt
{

namespace graphlib
{
class Graph;
}

enum class GraphType : std::uint8_t
{
    Forward = 0,
    Backward = 1,
    Loss = 2,
    Optimizer = 3,
    GraphTypeCount = 4,
};

template <typename T>
constexpr std::underlying_type_t<T> to_underlying(T e) noexcept
{
    return static_cast<std::underlying_type_t<T>>(e);
}

constexpr std::uint8_t GRAPH_TYPE_COUNT = to_underlying(GraphType::GraphTypeCount);
using StaticGraphArray = std::array<graphlib::Graph*, GRAPH_TYPE_COUNT>;

/**
 * @brief ForgeGraphModule is a container for all the graphs that are part of a module.
 * The graphs are stored in an array by their type (enum GraphType).
 * ForgeGraphModule is initialized with a Forward graph,
 * while the other graphs can be set later (if the module is compiled for training).
 */
class ForgeGraphModule
{
   public:
    ForgeGraphModule(std::string name, graphlib::Graph* forward_graph) : name_(name), graphs_{nullptr}
    {
        TT_ASSERT(forward_graph != nullptr);
        graphs_[to_underlying(GraphType::Forward)] = forward_graph;
    }

    void set_graph(GraphType type, graphlib::Graph* graph)
    {
        TT_ASSERT(graph != nullptr);
        graphs_[to_underlying(type)] = graph;
    }

    graphlib::Graph* get_graph(GraphType type) const { return graphs_[to_underlying(type)]; }

    /**
     * @brief Get all existing graphs in the module.
     * @return A vector of pointers to the graphs.
     */
    std::vector<graphlib::Graph*> graphs() const
    {
        std::vector<graphlib::Graph*> res;
        res.reserve(graphs_.size());
        for (auto graph : graphs_)
        {
            if (graph != nullptr)
            {
                res.push_back(graph);
            }
        }
        return res;
    }

    std::string name() const { return name_; }

   private:
    std::string name_;

    // Static array of graphs, indexed by GraphType.
    StaticGraphArray graphs_;
};

}  // namespace tt
