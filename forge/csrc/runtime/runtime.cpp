// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime.hpp"

#include <optional>

#include "graph_lib/graph.hpp"
#include "tensor.hpp"
#include "tt/runtime/runtime.h"
#include "tt_device.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt
{

template <typename T>
std::vector<int64_t> as_vec_int64(std::vector<T> const& vec)
{
    std::vector<int64_t> result;
    result.reserve(vec.size());
    for (auto const& v : vec)
    {
        result.push_back(v);
    }
    return result;
}

static runtime::Tensor create_tensor(const torch::Tensor& tensor)
{
    auto data = std::shared_ptr<void>(
        tensor.data_ptr(),
        [tensor](void*) { (void)tensor; }  // Capture tensor by value to increase ref count and keep it alive
    );

    auto shape = std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
    auto stride = std::vector<uint32_t>(tensor.strides().begin(), tensor.strides().end());

    return runtime::createTensor(
        data, shape, stride, tensor.element_size(), torch_scalar_type_to_dt(tensor.scalar_type()));
}

runtime::Binary load_binary_from_file(std::string const& filename)
{
    runtime::Binary binary = tt::runtime::Binary::loadFromPath(filename.c_str()).handle;
    return binary;
}

std::vector<torch::Tensor> run_binary_from_file(
    std::string const& filename, int program_idx, std::vector<torch::Tensor> const& inputs)
{
    auto binary = load_binary_from_file(filename);

    return run_binary(binary, program_idx, inputs);
}

void verify_input_tensors(
    const std::vector<torch::Tensor>& input_tensors, const std::vector<runtime::TensorDesc>& input_descs)
{
    if (input_tensors.size() != input_descs.size())
    {
        log_fatal(LogTTDevice, "Input count mismatch: expected {}, got {}", input_descs.size(), input_tensors.size());
    }

    for (size_t i = 0; i < input_descs.size(); ++i)
    {
        const auto& input_tensor = input_tensors[i];
        const auto& desc = input_descs[i];

        auto shape = as_vec_int64(desc.shape);
        // auto stride = as_vec_int64(desc.stride);

        if (input_tensor.sizes().vec() != shape)
        {
            log_fatal(
                LogTTDevice, "Tensor {} - shape mismatch: expected {}, got {}", i, shape, input_tensor.sizes().vec());
        }

        // if (input_tensor.strides().vec() != stride)
        // {
        //     log_fatal(
        //         LogTTDevice,
        //         "Tensor {} - stride mismatch: expected {}, got {}",
        //         i,
        //         stride,
        //         input_tensor.strides().vec());
        // }
        //
        if (torch_scalar_type_to_dt(input_tensor.scalar_type()) != desc.dataType)
        {
            auto expected = target::EnumNameDataType(desc.dataType);
            auto got = target::EnumNameDataType(torch_scalar_type_to_dt(input_tensor.scalar_type()));
            log_fatal(LogTTDevice, "Tensor {} - data type mismatch: expected {}, got {}", i, expected, got);
        }
    }
}

void verify_tensor(size_t idx, const torch::Tensor& input, const runtime::TensorDesc& desc)
{
    auto shape = as_vec_int64(desc.shape);
    auto stride = as_vec_int64(desc.stride);

    if (input.sizes().vec() != shape)
    {
        log_fatal(LogTTDevice, "Tensor {} - shape mismatch: expected {}, got {}", idx, shape, input.sizes().vec());
    }

    if (input.strides().vec() != stride)
    {
        log_fatal(LogTTDevice, "Tensor {} - stride mismatch: expected {}, got {}", idx, stride, input.strides().vec());
    }

    if (torch_scalar_type_to_dt(input.scalar_type()) != desc.dataType)
    {
        auto expected = target::EnumNameDataType(desc.dataType);
        auto got = target::EnumNameDataType(torch_scalar_type_to_dt(input.scalar_type()));
        log_fatal(LogTTDevice, "Tensor {} - data type mismatch: expected {}, got {}", idx, expected, got);
    }
}

// void verify_input_tensors(
//     const std::vector<Tensor>& input_tensors, const std::vector<runtime::TensorDesc>& input_descs)
// {
//     if (input_tensors.size() != input_descs.size())
//     {
//         log_fatal(LogTTDevice, "Input count mismatch: expected {}, got {}", input_descs.size(),
//         input_tensors.size());
//     }
//
//     for (size_t i = 0; i < input_descs.size(); ++i)
//     {
//         const auto& input_tensor = input_tensors[i];
//         const auto& desc = input_descs[i];
//
//         verify_tensor(i, input_tensor.storage(), desc);
//     }
// }
//
std::vector<Tensor> construct_persistent_tensors(const graphlib::Graph* graph, const TensorPool& tensor_pool)
{
    auto const_names = graph->get_constant_names();
    auto param_names = graph->get_parameter_names();

    // Reserve space for module inputs, constants and parameters.
    std::vector<Tensor> inputs;
    inputs.reserve(const_names.size() + param_names.size());

    // Append constant and param tensors.
    std::vector<std::string> params_and_consts_names;
    params_and_consts_names.insert(params_and_consts_names.end(), const_names.begin(), const_names.end());
    params_and_consts_names.insert(params_and_consts_names.end(), param_names.begin(), param_names.end());

    for (auto const& name : params_and_consts_names)
    {
        inputs.push_back(tensor_pool.get_tensor(name));
    }

    return inputs;
}

std::vector<tt::Tensor> create_runtime_tensors(std::vector<torch::Tensor> tensors)
{
    std::vector<tt::Tensor> rt_tensors;
    rt_tensors.reserve(tensors.size());

    for (auto& tensor : tensors)
    {
        rt_tensors.emplace_back(tensor);
    }

    return rt_tensors;
}

std::vector<torch::Tensor> run_binary(
    runtime::Binary& binary,
    int program_idx,
    std::vector<torch::Tensor> const& act_inputs,
    std::vector<tt::Tensor> persistent_inputs)
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
    auto& tt_device = system.devices[0];
    if (!tt_device->is_open())
    {
        log_fatal(LogTTDevice, "Failed to open device");
    }

    auto& device = *tt_device->rt_device;

    auto input_descs = binary.getProgramInputs(program_idx);

    std::vector<runtime::Tensor> inputs;
    inputs.reserve(act_inputs.size() + persistent_inputs.size());

    for (auto& tensor : act_inputs)
    {
        inputs.push_back(create_tensor(tensor));
    }

    for (auto& tensor : persistent_inputs)
    {
        inputs.push_back(tensor.get_runtime_tensor());
    }

    // verify_input_tensors(inputs, input_descs);

    std::vector<torch::Tensor> outputs;
    std::vector<runtime::Tensor> rt_outputs;
    std::vector<runtime::TensorDesc> output_descs = binary.getProgramOutputs(program_idx);
    outputs.reserve(output_descs.size());
    for (auto const& desc : output_descs)
    {
        std::vector<std::int64_t> shape = as_vec_int64(desc.shape);
        // std::vector<std::int64_t> stride = as_vec_int64(desc.stride);
        std::vector<std::int64_t> stride(shape.size(), 1);
        for (size_t i = shape.size() - 2; i >= 0; --i)
        {
            stride[i] = shape[i + 1] * stride[i + 1];
        }

        torch::Tensor output = at::empty_strided(shape, stride, dt_to_torch_scalar_type(desc.dataType));
        outputs.emplace_back(std::move(output));
        rt_outputs.emplace_back(create_tensor(outputs.back()));
    }

    std::vector<runtime::Tensor> submit_outputs = runtime::submit(device, binary, program_idx, inputs);
    TT_ASSERT(submit_outputs.size() == rt_outputs.size(), "Output count mismatch");
    for (size_t i = 0; i < submit_outputs.size(); ++i)
    {
        runtime::memcpy(rt_outputs[i], submit_outputs[i]);
        runtime::deallocateTensor(submit_outputs[i], true);
    }

    return outputs;
}

std::vector<torch::Tensor> run_binary(
    runtime::Binary& binary, int program_idx, std::vector<torch::Tensor> const& inputs)
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
    auto& tt_device = system.devices[0];
    if (!tt_device->is_open())
    {
        log_fatal(LogTTDevice, "Failed to open device");
    }

    auto& device = *tt_device->rt_device;

    auto input_descs = binary.getProgramInputs(program_idx);
    verify_input_tensors(inputs, input_descs);

    std::vector<runtime::Tensor> rt_inputs;
    size_t input_idx = 0;
    for (auto const& input : inputs)
    {
        auto tensor = create_tensor(input);
        auto layout = tt::runtime::getLayout(binary, program_idx, input_idx++);
        tt::runtime::toLayout(tensor, device, layout);
        rt_inputs.emplace_back(tensor);
    }

    std::vector<torch::Tensor> outputs;
    std::vector<runtime::Tensor> rt_outputs;
    std::vector<runtime::TensorDesc> output_descs = binary.getProgramOutputs(program_idx);
    outputs.reserve(output_descs.size());
    for (auto const& desc : output_descs)
    {
        std::vector<std::int64_t> shape = as_vec_int64(desc.shape);
        // std::vector<std::int64_t> stride = as_vec_int64(desc.stride);
        std::vector<std::int64_t> stride(shape.size());
        stride[shape.size() - 1] = 1;
        for (size_t i = shape.size() - 2; i >= 0 && i < shape.size(); --i)
        {
            stride[i] = shape[i + 1] * stride[i + 1];
        }

        torch::Tensor output = at::empty_strided(shape, stride, dt_to_torch_scalar_type(desc.dataType));
        outputs.emplace_back(std::move(output));
        rt_outputs.emplace_back(create_tensor(outputs.back()));
    }

    std::vector<runtime::Tensor> submit_outputs = runtime::submit(device, binary, program_idx, rt_inputs);
    TT_ASSERT(submit_outputs.size() == rt_outputs.size(), "Output count mismatch");
    for (size_t i = 0; i < submit_outputs.size(); ++i)
    {
        auto host = runtime::toHost(submit_outputs[i], true /*untilize*/);
        runtime::memcpy(rt_outputs[i], host);
        runtime::deallocateTensor(submit_outputs[i], true);
    }

    return outputs;
}

}  // namespace tt
