// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime.hpp"
#include <optional>

#include "tt_device.hpp"
#include "utils/logger.hpp"
#include "tt/runtime/runtime.h"

namespace tt {

static target::DataType torch_scalar_type_to_dt(torch::ScalarType st)
{
    switch (st)
    {
        case torch::ScalarType::Byte: return target::DataType::UInt8;
        case torch::ScalarType::Char: return target::DataType::UInt8;
        case torch::ScalarType::Short: return target::DataType::UInt16;
        case torch::ScalarType::Int: return target::DataType::UInt32;
        case torch::ScalarType::Long: return target::DataType::UInt32;
        case torch::ScalarType::Half: return target::DataType::Float16;
        case torch::ScalarType::Float: return target::DataType::Float32;
        // case torch::ScalarType::Double:
        // case torch::ScalarType::ComplexHalf:
        // case torch::ScalarType::ComplexFloat:
        // case torch::ScalarType::ComplexDouble:
        // case torch::ScalarType::Bool:
        case torch::ScalarType::BFloat16: return target::DataType::BFloat16;
        default: break;
    }

    log_fatal(LogTTDevice, "Unhandled dtype {}", st);
}

static torch::ScalarType dt_to_torch_scalar_type(target::DataType df)
{
    switch (df)
    {
        case target::DataType::UInt8: return torch::ScalarType::Byte;
        case target::DataType::UInt16: return torch::ScalarType::Short;
        case target::DataType::UInt32: return torch::ScalarType::Int;
        case target::DataType::Float16: return torch::ScalarType::Half;
        case target::DataType::Float32: return torch::ScalarType::Float;
        case target::DataType::BFloat16: return torch::ScalarType::BFloat16;
        default: break;
    }
    
    log_fatal(LogTTDevice, "Unhandled dtype {}", target::EnumNameDataType(df));
}

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
        data,
        shape,
        stride,
        tensor.element_size(),
        torch_scalar_type_to_dt(tensor.scalar_type()));
}

runtime::Binary load_binary_from_file(std::string const& filename)
{
    runtime::Binary binary = tt::runtime::Binary::loadFromPath(filename.c_str()).handle;
    return binary;
}

std::vector<torch::Tensor> run_binary_from_file(std::string const& filename, int program_idx, std::vector<torch::Tensor> const& inputs)
{
    auto binary = load_binary_from_file(filename);

    return run_binary(binary, program_idx, inputs);
}

std::vector<torch::Tensor> run_binary(runtime::Binary &binary, int program_idx, std::vector<torch::Tensor> const& inputs)
{
    auto& system = TTSystem::get_system();

    for (auto &device : system.devices)
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

    std::vector<runtime::Tensor> rt_inputs;
    for (auto const& input : inputs)
    {
        rt_inputs.emplace_back(create_tensor(input));
    }

    std::vector<torch::Tensor> outputs;
    std::vector<runtime::Tensor> rt_outputs;
    std::vector<runtime::TensorDesc> output_descs = binary.getProgramOutputs(program_idx);
    outputs.reserve(output_descs.size());
    for (auto const& desc : output_descs)
    {
        std::vector<std::int64_t> shape = as_vec_int64(desc.shape);
        std::vector<std::int64_t> stride = as_vec_int64(desc.stride);

        torch::Tensor output = at::empty_strided(shape, stride, dt_to_torch_scalar_type(desc.dataType));
        outputs.emplace_back(std::move(output));
        rt_outputs.emplace_back(create_tensor(outputs.back()));
    }

    runtime::Event _ = runtime::submit(device, binary, program_idx, rt_inputs, rt_outputs);

    return outputs;
}

} // namespace tt
