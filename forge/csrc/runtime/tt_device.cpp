// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_device.hpp"

#include <optional>

#include "tt/runtime/runtime.h"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt
{

static bool system_is_initialized = false;

TTSystem detect_available_devices()
{
    auto [system_desc, chip_ids] = runtime::getCurrentSystemDesc();

    std::vector<std::shared_ptr<TTDevice>> devices;
    int logical_device_index = 0;
    ARCH arch = ARCH::Invalid;
    for (std::uint32_t chip_desc_index : *system_desc->chip_desc_indices())
    {
        target::ChipDesc const* chip_desc = system_desc->chip_descs()->Get(chip_desc_index);
        target::ChipCapability chip_capabilities = system_desc->chip_capabilities()->Get(logical_device_index);

        bool mmio = bool(chip_capabilities & target::ChipCapability::HostMMIO);
        if (not mmio)
        {
            continue;
        }

        switch (chip_desc->arch())
        {
            case target::Arch::Grayskull: arch = ARCH::GRAYSKULL; break;
            case target::Arch::Wormhole_b0: arch = ARCH::WORMHOLE_B0; break;
            case target::Arch::Blackhole: arch = ARCH::BLACKHOLE; break;
            default: log_fatal(LogTTDevice, "Unknown chip type {}", target::EnumNameArch(chip_desc->arch()));
        }

        auto device = std::make_shared<TTDevice>(std::nullopt, system_desc, arch, mmio, logical_device_index);
        devices.push_back(device);
        ++logical_device_index;
    }

    system_is_initialized = true;
    return TTSystem{system_desc, chip_ids, devices};
}

TTSystem& TTSystem::get_system()
{
    static TTSystem system = detect_available_devices();
    return system;
}

bool TTSystem::is_initialized() { return system_is_initialized; }

void TTDevice::open_device()
{
    TT_ASSERT(!is_open());
    rt_device = runtime::openDevice({index});
}

void TTDevice::close_device()
{
    TT_ASSERT(is_open());
    runtime::closeDevice(rt_device.value());
    rt_device.reset();
}

}  // namespace tt
