// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace at
{
class Tensor;  // Forward declaration of torch tensor.
}

namespace tt
{
namespace graphlib
{
class Shape;
struct NodeContext;
using DimBroadcastTrampoline = std::tuple<int, int, int>;
}  // namespace graphlib

namespace autograd
{
struct autograd_context;
}

class DecomposingContext;

namespace ops
{
class Op;

/**
 * Declaration for ops interface in a separate namespace ns.
 */
#define DECLARE_OP_INTERFACE(ns)                                                                      \
    namespace ns                                                                                      \
    {                                                                                                 \
    at::Tensor eval(const Op &op, const std::vector<at::Tensor> &tensors);                            \
                                                                                                      \
    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(                 \
        const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs);                         \
                                                                                                      \
    tt::graphlib::NodeContext backward(                                                               \
        const Op &op,                                                                                 \
        tt::autograd::autograd_context &context,                                                      \
        int operand,                                                                                  \
        const std::vector<tt::graphlib::NodeContext> &inputs,                                         \
        const tt::graphlib::NodeContext &output,                                                      \
        const tt::graphlib::NodeContext &gradient);                                                   \
                                                                                                      \
    void decompose(                                                                                   \
        const Op &op,                                                                                 \
        const char *dispatch,                                                                         \
        DecomposingContext &dc,                                                                       \
        const std::vector<tt::graphlib::NodeContext> &inputs);                                        \
                                                                                                      \
    long initial_flops_estimate(const Op &op, const std::vector<std::vector<std::uint32_t>> &inputs); \
    }

/**
 * Ops declaration.
 */
DECLARE_OP_INTERFACE(abs);
DECLARE_OP_INTERFACE(constant);

#undef DECLARE_OP_INTERFACE

}  // namespace ops
}  // namespace tt
