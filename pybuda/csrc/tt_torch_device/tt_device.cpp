// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "pybuda/csrc/tt_torch_device/tt_device.hpp"

#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pybuda/csrc/lower_to_buda/common.hpp"
#include "tt/runtime/runtime.h"
#include "utils/assert.hpp"
#include "utils/env.hpp"
#include "utils/logger.hpp"

namespace tt
{
struct RunPrograms
{
    std::vector<Program> programs;
    std::vector<torch::Tensor> inputs;
    std::unordered_map<std::string, torch::Tensor> parameters;
};

struct Barrier
{
};

using Command = std::variant<Barrier, RunPrograms>;

struct CommandQueue
{
    Workload* workload = nullptr;
    std::vector<Command> commands;
};

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

    log_fatal(LogTTDevice, "Unhandled dtype {}", df);
}

void pad_to_buda_shape(torch::Tensor & tensor)
{
    auto tt_device = tensor.device();
    auto cpu_tensor = tensor.to(torch::kCPU);
    if (cpu_tensor.sizes().size() > 4) {
        throw std::runtime_error("Tensor has more than 4 dimensions");
    } else if (cpu_tensor.sizes().size() < 4) {
        auto tensor_impl = cpu_tensor.unsafeGetTensorImpl();
        std::vector<int64_t> new_shape;
        for (size_t i = 0; i < cpu_tensor.sizes().size(); i++) {
            new_shape.push_back(cpu_tensor.sizes()[i]);
        }
        while (new_shape.size() < 4) {
            new_shape.insert(new_shape.begin(), 1);
        }
        tensor_impl->Reshape(new_shape);
    }
    auto new_shape = cpu_tensor.sizes();
    namespace F = torch::nn::functional;
    cpu_tensor = torch::nn::functional::pad(
        cpu_tensor, 
        F::PadFuncOptions(
            {0, align_up_tile(new_shape[3]) - new_shape[3],
             0, align_up_tile(new_shape[2]) - new_shape[2]}
        ).mode(torch::kConstant));

    cpu_tensor.unsafeGetTensorImpl()->set_size(2, align_up_tile(new_shape[2]));
    cpu_tensor.unsafeGetTensorImpl()->set_size(3, align_up_tile(new_shape[3]));

    int64_t curr_stride = 1;

    for (int i = 3; i >= 0; i--) {
        cpu_tensor.unsafeGetTensorImpl()->set_stride(i, curr_stride);
        curr_stride *= cpu_tensor.sizes()[i];
    }

    tensor = cpu_tensor.to(tt_device);
}

std::vector<std::uint32_t> fromIntArrayRef(torch::IntArrayRef arr)
{
    std::vector<std::uint32_t> vec;
    for (auto i : arr)
        vec.push_back(i);
    return vec;
}

runtime::Tensor create_tensor(const torch::Tensor& tensor)
{
    auto data = std::shared_ptr<void>(
        tensor.data_ptr(),
        [tensor](void*) { (void)tensor; }  // Capture tensor by value to increase ref count and keep it alive
    );
    return runtime::createTensor(
        data,
        fromIntArrayRef(tensor.sizes()),
        fromIntArrayRef(tensor.strides()),
        tensor.element_size(),
        torch_scalar_type_to_dt(tensor.scalar_type()));
}

template <typename T>
std::vector<std::int64_t> asInt64Vec(std::vector<T> const& v)
{
    std::vector<std::int64_t> result;
    result.reserve(v.size());
    for (auto const& i : v)
        result.push_back(i);
    return result;
}

std::vector<torch::Tensor> dispatch(
    TTDevice& device,
    std::shared_ptr<Workload> workload,
    int program_idx,
    std::vector<torch::Tensor>& inputs,
    bool const& is_compile)
{
    int input_idx = 0;
    std::vector<runtime::Tensor> rt_inputs;
    rt_inputs.reserve(workload->inputs.at(program_idx).size());
    for ([[ maybe_unused ]] auto const& desc : workload->inputs.at(program_idx))
    {
        torch::Tensor & input = inputs.at(input_idx);
        auto impl = input.unsafeGetTensorImpl();
        TTMetaData *input_meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());

        TT_ASSERT (input_meta != nullptr);
        if (!input_meta->runtime_transformed and !input_meta->created_on_device)
        {
            std::string runtime_transform = device.input_runtime_transforms.at(program_idx).at(input_idx);
            std::vector<int> tile_bcast_dims = device.input_tile_bcast_dims.at(program_idx).at(input_idx);
            auto transformed_input = eval_runtime_transform(input.to(torch::kCPU), runtime_transform, tile_bcast_dims);
            input_meta->runtime_transformed = true;
            rt_inputs.emplace_back(create_tensor(transformed_input));
        }
        else
        {
            rt_inputs.emplace_back(create_tensor(input));
        }
        ++input_idx;
    }

    runtime::Binary binary(workload->flatbuffer);
    std::vector<torch::Tensor> outputs;
    std::vector<runtime::Tensor> rt_outputs;
    std::vector<runtime::TensorDesc> output_descs = binary.getProgramOutputs(program_idx);
    outputs.reserve(output_descs.size());
    for (auto const& desc : output_descs)
    {
        std::vector<std::int64_t> shape = asInt64Vec(desc.shape);
        std::vector<std::int64_t> stride = asInt64Vec(desc.stride);
        outputs.emplace_back(empty_strided(shape, stride, dt_to_torch_scalar_type(desc.dataType)));
        rt_outputs.emplace_back(create_tensor(outputs.back()));
    }

    runtime::Event event = runtime::submit(device.rt_device, binary, program_idx, rt_inputs, rt_outputs);
    (void)event;

    // Clear old tensor uids and update with new ones
    if (device.subgraph_to_tensor_uid_on_device.count(program_idx) != 0)
        device.subgraph_to_tensor_uid_on_device[program_idx].clear();

    int output_idx = 0;
    const auto& subgraph_outputs = workload->outputs.at(program_idx);
    for (auto const& output : outputs)
    {
        PyBudaTensorDesc const& desc = subgraph_outputs.at(output_idx );

        std::string runtime_transform = device.output_runtime_transforms.at(program_idx).at(output_idx );
        // auto impl = output.unsafeGetTensorImpl();
        // auto output_tensor_uid = dynamic_cast<TTMetaData*>(impl->get_backend_meta())->unique_output_id;

        // if (queue_desc.io_type == IO_TYPE::RandomAccess) {
        //     register_output_runtime_transform(output, runtime_transform);
        //     device.subgraph_to_tensor_uid_on_device[program_idx].push_back(output_tensor_uid);
        //     outputs.emplace_back(output);
        // } else 
        {
            PyGILState_STATE gstate=PyGILState_Ensure();
            auto tt_device_ = output.device();
            // Move tensor to CPU because torch::narrow is only supported on CPU for now
            torch::Tensor cpu_output = output.to(
            torch::kCPU, output.scalar_type(), false, true);
            register_output_runtime_transform(output, runtime_transform);

            for (size_t i = 0; i < cpu_output.sizes().size(); i++)
            {
                if (cpu_output.sizes()[i] != desc.shape[i]) {
                    log_trace(LogTorchDevice, "narrowing dim[{}] start[{}] length[{}]", i, 0, desc.shape[i]);
                    cpu_output = torch::narrow(cpu_output, i, 0, desc.shape[i]);
                }
            }
            // Move tensor back to TT device
            // (TODO: this is a workaround, we should be able to do this without calling contiguous, which makes a copy)
            torch::Tensor tt_output_ = cpu_output.contiguous().to(
                tt_device_, cpu_output.scalar_type(), false, false/* copy */);
            PyGILState_Release(gstate);
            outputs.emplace_back(tt_output_);
        }
        ++output_idx;
    }
    return outputs;
}

std::vector<TTDevice> query_available_tt_devices()
{
    static std::shared_ptr<TTContext> context = std::make_shared<TTContext>();
    std::vector<TTDevice> d;

    auto [system_desc, device_ids] = runtime::getCurrentSystemDesc();

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
        switch(chip_desc->arch())
        {
            case target::Arch::Grayskull: arch = ARCH::GRAYSKULL; break;
            case target::Arch::Wormhole_b0: arch = ARCH::WORMHOLE_B0; break;
            case target::Arch::Blackhole: arch = ARCH::BLACKHOLE; break;
            default: log_fatal(LogTTDevice, "Unknown chip type {}", chip_desc->arch());
        }
        ++logical_device_index;
    }

    if (arch == ARCH::Invalid)
        log_fatal(LogTTDevice, "No available devices detected (To run with golden device, set PYBUDA_DEVMODE=1)");

    runtime::Device rt_device = runtime::openDevice(device_ids);
    d.emplace_back(rt_device, system_desc, device_ids, arch, true, 0, context);

    log_debug(LogTTDevice, "Available devices:");
    for (int i = 0; i < (int)d.size(); ++i) log_debug(LogTTDevice, "  [{}] {}", i, d[i].arch);
    return d;
}

std::string get_device_cluster_yaml(TTDevice const&) { return "";} //TODO }

std::string to_string(TTDevice const& d)
{
    return device_type_name(TT, true /*lower_case*/) + ":" + std::to_string(d.index);
}

torch::Device torch_device(TTDevice const& d) { return torch_device_at_index(d.index); }

TTContext::~TTContext() {;}

torch::Tensor eval_runtime_transform(
    const torch::Tensor& tensor,
    std::string transform,
    std::vector<int> &tile_bcast_dims)
{
    py::object py_tensor = py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));

    PyGILState_STATE gstate=PyGILState_Ensure();
    auto module = py::module_::import("pybuda.tensor");
    py::function eval_transform = module.attr("eval_runtime_transform");
    py::tuple py_result = eval_transform(transform, py_tensor, tile_bcast_dims);
    PyGILState_Release(gstate);
    torch::Tensor torch_tensor = THPVariable_Unpack(static_cast<PyObject *>(py_result[0].ptr()));
    return torch_tensor;
}

torch::Tensor narrow_to_pytorch(const torch::Tensor& tensor, std::string transform)
{
    //TODO
    py::object py_tensor = py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));

    PyGILState_STATE gstate=PyGILState_Ensure();
    auto module = py::module_::import("pybuda.tensor");
    py::function eval_transform = module.attr("eval_runtime_transform"); //TODO: update
    py::object py_result = eval_transform(transform, py_tensor);
    PyGILState_Release(gstate);
    torch::Tensor torch_tensor = THPVariable_Unpack(static_cast<PyObject *>(py_result.ptr()));
    return torch_tensor;
}

bool is_created_on_device(const torch::Tensor& tensor)
{
    auto impl = tensor.unsafeGetTensorImpl();
    TTMetaData* meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());
    TT_ASSERT(meta != nullptr);
    return meta->created_on_device;
}

std::vector<size_t> original_shape(const torch::Tensor& tensor)
{
    auto impl = tensor.unsafeGetTensorImpl();
    TTMetaData* meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());
    TT_ASSERT(meta != nullptr);
    std::vector<size_t> shape;
    for (auto s : meta->original_shape)
        shape.push_back(s);

    return shape;
}
int unique_id(const torch::Tensor& tensor)
{
    auto impl = tensor.unsafeGetTensorImpl();
    TTMetaData* meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());
    if (meta != nullptr)
        return meta->unique_output_id;

    return -1;
}

}  // namespace tt
