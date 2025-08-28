// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "reportify/to_json.hpp"

#include "nlohmann/json.hpp"
#include "ops/op.hpp"

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

void to_json(json& j, EdgeAttributes const& attrs)
{
    j["ublock_order"] = attrs.get_ublock_order();

    std::vector<std::string> str_tms;
    for (const ops::Op& op : attrs.get_tms()) str_tms.push_back(op.as_string());

    j["tms"] = str_tms;
}
}  // namespace graphlib

namespace ops
{
void to_json(json& j, ops::Op const& op)
{
    j["op"] = {};
    j["op"]["type"] = op.as_string();
    j["op"]["attrs"] = op.attrs();
}

}  // namespace ops
}  // namespace tt
