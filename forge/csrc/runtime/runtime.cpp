// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime.hpp"

#include <optional>

#include "tensor.hpp"
#include "tt/runtime/runtime.h"
#include "tt/runtime/ttnn/test/dylib.h"
#include "tt_device.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt
{

runtime::Binary load_binary_from_file(std::string const& filename)
{
    runtime::Binary binary = tt::runtime::Binary::loadFromPath(filename.c_str()).handle;
    return binary;
}

std::vector<tt::Tensor> run_program_from_file(
    std::string const& filename, int program_idx, std::vector<torch::Tensor>& inputs)
{
    auto binary = load_binary_from_file(filename);

    std::vector<tt::Tensor> tt_inputs;
    tt_inputs.reserve(inputs.size());

    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(tt_inputs),
        [](torch::Tensor& input) { return tt::Tensor(input); });

    return run_program(binary, program_idx, tt_inputs);
}

void verify_input_descs(
    const std::vector<runtime::TensorDesc>& descs, const std::vector<runtime::TensorDesc>& expected_descs)
{
    if (descs.size() != expected_descs.size())
    {
        log_fatal(LogTTDevice, "Input count mismatch: expected {}, got {}", descs.size(), expected_descs.size());
    }

    for (size_t i = 0; i < descs.size(); ++i)
    {
        const auto& desc = descs[i];
        const auto& expected_desc = expected_descs[i];

        if (desc.shape != expected_desc.shape)
        {
            log_fatal(
                LogTTDevice, "Tensor {} - shape mismatch: expected {}, got {}", i, expected_desc.shape, desc.shape);
        }

        if (desc.stride != expected_desc.stride)
        {
            log_fatal(
                LogTTDevice, "Tensor {} - stride mismatch: expected {}, got {}", i, expected_desc.stride, desc.stride);
        }

        if (desc.dataType != expected_desc.dataType)
        {
            auto expected = target::EnumNameDataType(expected_desc.dataType);
            auto got = target::EnumNameDataType(desc.dataType);
            log_fatal(LogTTDevice, "Tensor {} - data type mismatch: expected {}, got {}", i, expected, got);
        }
    }
}

std::vector<tt::Tensor> run_program(runtime::Binary& binary, int program_idx, std::vector<tt::Tensor>& inputs)
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

    auto expected_input_descs = binary.getProgramInputs(program_idx);

    std::vector<runtime::TensorDesc> input_descs;
    input_descs.reserve(inputs.size());

    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(input_descs),
        [](const tt::Tensor& input) { return input.tensor_desc(); });

    verify_input_descs(input_descs, expected_input_descs);

    std::vector<runtime::Tensor> rt_inputs;
    rt_inputs.reserve(inputs.size());

    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(rt_inputs),
        [](tt::Tensor& input)
        {
            runtime::Tensor& tensor = input.get_runtime_tensor();
            runtime::setTensorRetain(tensor, /*retain=*/true);
            return tensor;
        });

    auto output_descs = binary.getProgramOutputs(program_idx);
    std::vector<runtime::Tensor> rt_outputs = runtime::submit(device, binary, program_idx, rt_inputs);
    TT_ASSERT(output_descs.size() == rt_outputs.size(), "Output count mismatch");

    std::vector<tt::Tensor> outputs;
    outputs.reserve(rt_outputs.size());

    size_t output_id = 0;
    std::transform(
        rt_outputs.begin(),
        rt_outputs.end(),
        std::back_inserter(outputs),
        [&output_id, &output_descs](runtime::Tensor& rt_output)
        {
            auto desc = output_descs[output_id++];
            return tt::Tensor(rt_output, desc);
        });

    return outputs;
}

void* open_so(std::string path) { return runtime::ttnn::test::openSo(path); }
void close_so(void* handle) { return runtime::ttnn::test::closeSo(handle); }

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
        runtime::ttnn::test::runSoProgram(so_handle, func_name, rt_inputs_consts_params, device);

    return rt_outputs;
}

bool compareOuts(std::vector<tt::runtime::Tensor>& lhs, std::vector<tt::runtime::Tensor>& rhs)
{
    return runtime::ttnn::test::compareOuts(lhs, rhs);
}

}  // namespace tt
