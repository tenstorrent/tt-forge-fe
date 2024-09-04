// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "reportify/to_json.hpp"

#include "nlohmann/json.hpp"

namespace tt
{
namespace graphlib
{
void to_json(json& j, UBlockOrder const& ublock_order)
{
    switch (ublock_order)
    {
        case UBlockOrder::R: j = "R"; break;
        case UBlockOrder::C: j = "C"; break;
        default: j = "UBlockOrder::Unknown"; break;
    }
}

void to_json(json& j, OpType const& op_type)
{
    j["op_type"] = {};
    j["op_type"]["type"] = op_type.op;
    j["op_type"]["attrs"] = op_type.attr;
    j["op_type"]["forge_attrs"] = op_type.forge_attrs;
    j["op_type"]["named_attrs"] = op_type.named_attrs;
}

void to_json(json& j, EdgeAttributes const& attrs)
{
    j["ublock_order"] = attrs.get_ublock_order();
    j["tms"] = attrs.get_tms();
}
}  // namespace graphlib
}  // namespace tt
