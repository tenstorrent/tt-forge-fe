// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>

#include "backend_api/device_config.hpp"
#include "utils/assert.hpp"

namespace tt
{
template <class T>
constexpr std::false_type false_type_t{};

template <typename T>
T DeviceConfig::get(std::string const &param, const bool system_level_command) const
{
    TT_ASSERT(false);
    if constexpr (std::is_same_v<T, CoreCoord>)
    {
        return CoreCoord(1, 1);
    }
    else if constexpr (std::is_same_v<T, DeviceGrid>)
    {
        return DeviceGrid(1, 1);
    }
    else
    {
        return T();
    }
}

// explicit instantiations
template std::string DeviceConfig::get<std::string>(std::string const &, const bool) const;
template std::uint32_t DeviceConfig::get<std::uint32_t>(std::string const &, const bool) const;
template std::uint64_t DeviceConfig::get<std::uint64_t>(std::string const &, const bool) const;
template int DeviceConfig::get<int>(std::string const &, const bool) const;
template bool DeviceConfig::get<bool>(std::string const &, const bool) const;
template CoreCoord DeviceConfig::get<CoreCoord>(std::string const &, const bool) const;
template std::vector<int> DeviceConfig::get<std::vector<int>>(std::string const &, const bool) const;
template std::unordered_map<uint32_t, EthCoord> DeviceConfig::get<std::unordered_map<uint32_t, EthCoord>>(
    std::string const &, const bool) const;
template std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>>
DeviceConfig::get<std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>>>(
    std::string const &, const bool) const;

// temporarily added, until FE consumes a commit that includes equivalent parsing in BBE
std::unordered_map<std::string, std::string> load_cached_sys_param(std::string yaml_file)
{
    std::unordered_map<std::string, std::string> cache;
    return cache;
}

void DeviceConfig::load_system_level_params() { TT_ASSERT(false); }

std::unordered_map<std::uint32_t, std::uint32_t> DeviceConfig::get_harvested_cfg() const
{
    TT_ASSERT(false);
    return {};
}
}  // namespace tt
