// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>

#include "forge/csrc/backend_api/arch_type.hpp"
#include "tt/runtime/types.h"

namespace tt
{

struct TTDevice
{
    std::optional<runtime::Device> rt_device;
    ARCH arch;
    bool mmio;
    int index;

    // TODO: These don't seem to belong here
    std::map<int, std::vector<std::string>> input_runtime_transforms;
    std::map<int, std::vector<std::vector<int>>> input_tile_bcast_dims;
    std::map<int, std::vector<std::string>> output_runtime_transforms;
    std::unordered_map<int, std::vector<int>> subgraph_to_tensor_uid_on_device;

    TTDevice(
        std::optional<runtime::Device> rt_device, runtime::SystemDesc system_desc, ARCH arch, bool mmio, int index) :
        rt_device(rt_device), arch(arch), mmio(mmio), index(index)
    {
    }

    TTDevice(const TTDevice&) = delete;
    TTDevice& operator=(const TTDevice&) = delete;

    bool is_open() const { return rt_device.has_value(); }

    void open_device();
    void close_device();
};

struct TTSystem
{
    runtime::SystemDesc system_desc;
    std::vector<int> chip_ids;
    std::vector<std::shared_ptr<TTDevice>> devices;

    TTSystem(const TTSystem&) = delete;
    TTSystem& operator=(const TTSystem&) = delete;

    ~TTSystem() { close_devices(); }

    void close_devices()
    {
        for (auto& device : devices)
        {
            if (device->is_open())
            {
                device->close_device();
            }
        }
    }

    static TTSystem& get_system();
};

TTSystem detect_available_devices();

}  // namespace tt
