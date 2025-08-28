// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/node_types.hpp"
#include "lower_to_forge/common.hpp"
#include "nlohmann/json_fwd.hpp"
#include "ops/op.hpp"

namespace std
{
template <typename... Ts>
void to_json(json& j, variant<Ts...> const& v)
{
    visit([&j](auto&& elem) { j = elem; }, v);
}
}  // namespace std

namespace tt
{
namespace graphlib
{
void to_json(json& j, UBlockOrder const& ublock_order);
void to_json(json& j, EdgeAttributes const& attrs);
}  // namespace graphlib

namespace ops
{
void to_json(json& j, ops::Op const& op_type);
}
}  // namespace tt
