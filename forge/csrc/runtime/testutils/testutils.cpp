// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "testutils.hpp"

#include "runtime/tensor.hpp"
#include "runtime/tt_device.hpp"
#include "tt/runtime/test/ttnn/dylib.h"
#include "utils/logger.hpp"

namespace tt::runtime::testutils
{

void* open_so(std::string path) { return ::tt::runtime::test::ttnn::openSo(path); }

void close_so(void* handle) { return ::tt::runtime::test::ttnn::closeSo(handle); }

std::vector<tt::runtime::Tensor> run_so_program(
    void* so_handle, std::string func_name, std::vector<tt::Tensor>& inputs, std::vector<tt::Tensor>& consts_and_params)
{
    auto& system = TTSystem::get_system();
    for (auto& device : system.devices)
    {
        if (!device->is_open())
        {
            device->open_device();
        }
    }

    // For now, we only support a single device.
    constexpr size_t device_id = 0;
    auto& tt_device = system.devices[device_id];
    if (!tt_device->is_open())
    {
        log_fatal(LogTTDevice, "Failed to open device");
    }

    auto& device = *tt_device->rt_device;

    std::vector<runtime::Tensor> rt_inputs;
    rt_inputs.reserve(inputs.size());

    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(rt_inputs),
        [](tt::Tensor& input) { return input.get_runtime_tensor(); });

    std::vector<runtime::Tensor> rt_inputs_consts_params;
    rt_inputs_consts_params.reserve(rt_inputs.size() + consts_and_params.size());
    for (auto& input : rt_inputs)
    {
        rt_inputs_consts_params.push_back(input);
    }
    for (auto& const_and_param : consts_and_params)
    {
        rt_inputs_consts_params.push_back(const_and_param.get_runtime_tensor());
    }

    std::vector<tt::runtime::Tensor> rt_outputs =
        ::tt::runtime::test::ttnn::runSoProgram(so_handle, func_name, rt_inputs_consts_params, device);

    return rt_outputs;
}

bool compareOuts(std::vector<tt::runtime::Tensor>& lhs, std::vector<tt::runtime::Tensor>& rhs)
{
    return ::tt::runtime::test::ttnn::compareOuts(lhs, rhs);
}

bool test_so(
    std::string so_path,
    std::string func_name,
    std::vector<tt::Tensor>& act_inputs,
    std::vector<tt::Tensor>& consts_and_params,
    std::vector<tt::Tensor>& golden_outs)
{
    void* so_handle = tt::runtime::testutils::open_so(so_path);
    std::vector<tt::runtime::Tensor> outs =
        tt::runtime::testutils::run_so_program(so_handle, func_name, act_inputs, consts_and_params);

    std::vector<runtime::Tensor> host_outs;
    std::transform(
        outs.begin(),
        outs.end(),
        std::back_inserter(host_outs),
        [](tt::runtime::Tensor& output)
        {
            std::vector<runtime::Tensor> vec_host_t = tt::runtime::toHost(output);
            assert(vec_host_t.size() == 1 && "We expect only one host tensor");
            return vec_host_t.front();
        });

    std::vector<runtime::Tensor> golden_host_outs;
    golden_host_outs.reserve(golden_outs.size());

    std::transform(
        golden_outs.begin(),
        golden_outs.end(),
        std::back_inserter(golden_host_outs),
        [](tt::Tensor& output)
        {
            runtime::Tensor rt_t = output.get_runtime_tensor();
            std::vector<runtime::Tensor> vec_host_t = tt::runtime::toHost(rt_t);
            assert(vec_host_t.size() == 1 && "We expect only one host tensor");
            return vec_host_t.front();
        });

    tt::runtime::testutils::close_so(so_handle);

    return tt::runtime::testutils::compareOuts(host_outs, golden_host_outs);
}

}  // namespace tt::runtime::testutils
