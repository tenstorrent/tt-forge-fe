// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "pybuda/csrc/tt_torch_device/tt_device.hpp"

#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "utils/assert.hpp"
#include "utils/env.hpp"
#include "utils/logger.hpp"
#include "pybuda/csrc/lower_to_buda/common.hpp"

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

std::shared_ptr<Workload> compile(TTDevice& device, CompileRequest const& compile_request)
{
    std::shared_ptr<Workload> workload = std::make_shared<Workload>(
        compile_request.output_dir,
        compile_request.inputs,
        compile_request.constants,
        compile_request.parameters,
        compile_request.outputs);

    // Todo: add transforms per subgraph to torch device so we can eval on tensot.to("tt")
    // register__ordered_input_runtime_transforms(compile_request.input_runtime_transforms);
    device.input_runtime_transforms = compile_request.input_runtime_transforms;
    device.input_tile_bcast_dims = compile_request.input_tile_bcast_dims;
    device.output_runtime_transforms = compile_request.output_runtime_transforms;

    return workload;
}

static DataFormat torch_scalar_type_to_df(torch::ScalarType st)
{
    switch (st)
    {
        case torch::ScalarType::Byte: return DataFormat::Int8;
        case torch::ScalarType::Char: return DataFormat::Int8;
        case torch::ScalarType::Short: return DataFormat::UInt16;
        case torch::ScalarType::Int: return DataFormat::RawUInt32;
        case torch::ScalarType::Long: return DataFormat::RawUInt32;
        case torch::ScalarType::Half: return DataFormat::Float16;
        case torch::ScalarType::Float: return DataFormat::Float32;
        // case torch::ScalarType::Double:
        // case torch::ScalarType::ComplexHalf:
        // case torch::ScalarType::ComplexFloat:
        // case torch::ScalarType::ComplexDouble:
        // case torch::ScalarType::Bool:
        case torch::ScalarType::BFloat16: return DataFormat::Float16_b;
        default: break;
    }

    log_fatal(LogTTDevice, "Unhandled dtype {}", st);
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


std::unordered_set<std::string> pushed;
void push_tensor(
    PyBudaTensorDesc const& desc,
    torch::Tensor & tensor,
    std::string const& info,
    std::optional<int> ptr)
{

    log_debug(
        LogTTDevice,
        "Pushing tensor({})[{}][{}] to device[{}]",
        tensor.data_ptr(),
        desc.name,
        tensor.scalar_type(),
        tensor.device());

    // TTNN from torch
    // TTNN to device
}

static torch::Tensor pop_tensor(PyBudaTensorDesc const& desc)
{
    log_debug(LogTTDevice, "Popping tensor[{}]", desc.name);

    //TTNN to torch
    torch::Tensor ret;
    
    return ret;
}

std::vector<torch::Tensor> dispatch(
    TTDevice & device,
    std::shared_ptr<Workload> workload,
    std::vector<Program> const& programs,
    std::vector<torch::Tensor> & inputs,
    int subgraph_idx,
    bool const& is_compile)
{

    int input_idx = 0;
    for (auto const& desc : workload->inputs.at(subgraph_idx))
    {
        torch::Tensor & input = inputs.at(input_idx);
        auto impl = input.unsafeGetTensorImpl();
        TTMetaData *input_meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());

        TT_ASSERT (input_meta != nullptr);
        if (!input_meta->runtime_transformed and !input_meta->created_on_device)
        {
            std::string runtime_transform = device.input_runtime_transforms.at(subgraph_idx).at(input_idx);
            std::vector<int> tile_bcast_dims = device.input_tile_bcast_dims.at(subgraph_idx).at(input_idx);
            auto transformed_input = eval_runtime_transform(input.to(torch::kCPU), runtime_transform, tile_bcast_dims);
            input_meta->runtime_transformed = true;
            push_tensor(transformed_input, fmt::format("input[{}]", input_idx));
        }
        else
        {
            if ((!input_meta->created_on_device))
            {
                push_tensor(input, fmt::format("input[{}]", input_idx));
            } else { //if (!tensor_populated_on_device) {
                // Compile mode needs to push tensor to device
                if (!is_compile)
                {
                    log_fatal(LogTTDevice, "Tensor created on device but not populated on device: {}", desc.name);
                } 
                else {
                    // Assign ptr = 0 since we are reading from RAM
                    log_info(LogTTDevice, "Graph Linking: compile stage --- Push {} to device", desc.name);
                    push_tensor(input, fmt::format("input[{}]", input_idx), 0/* ptr */);
                }                
            }
        }
        ++input_idx;
    }

    // for (Program const& program : programs)
    // {
    //     auto status = device.backend->run_program(program.name, program.parameters);
    //     if (status != DEVICE_STATUS_CODE::Success)
    //         log_fatal(LogTTDevice, "Failed to run_program: {} {}", program.name, status);
    // }

    std::vector<torch::Tensor> outputs;
    const auto &subgraph_outputs = workload->outputs.at(subgraph_idx);
    outputs.reserve(subgraph_outputs.size());

    // Clear old tensor uids and update with new ones
    if (device.subgraph_to_tensor_uid_on_device.count(subgraph_idx) != 0)
        device.subgraph_to_tensor_uid_on_device[subgraph_idx].clear();

    for (size_t i = 0; i < subgraph_outputs.size(); ++i)
    {
        PyBudaTensorDesc const& desc = subgraph_outputs.at(i);

        torch::Tensor output = pop_tensor(desc);
        std::string runtime_transform = device.output_runtime_transforms.at(subgraph_idx).at(i);
        auto impl = output.unsafeGetTensorImpl();
        auto output_tensor_uid = dynamic_cast<TTMetaData*>(impl->get_backend_meta())->unique_output_id;

        // if (queue_desc.io_type == IO_TYPE::RandomAccess) {
        //     register_output_runtime_transform(output, runtime_transform);
        //     device.subgraph_to_tensor_uid_on_device[subgraph_idx].push_back(output_tensor_uid);
        //     outputs.emplace_back(output);
        // } else {
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
        // }
    }
    return outputs;
}

std::vector<TTDevice> query_available_tt_devices()
{
    static std::shared_ptr<TTContext> context = std::make_shared<TTContext>();
    std::vector<TTDevice> d;
    // auto available_devices = backend::get_device_descs_for_available_devices();
    // if (available_devices.empty())
    // {
    //     constexpr bool mmio = true;
        
    //     ARCH arch = ARCH::Invalid;
    //     if (env_as<bool>("GOLDEN_WORMHOLE_B0"))
    //     {
    //         arch = ARCH::WORMHOLE_B0;
    //     }
    //     else if (env_as<bool>("PYBUDA_GOLDEN_BLACKHOLE"))
    //     {
    //         arch = ARCH::BLACKHOLE;
    //     }
    //     else {
    //         arch = ARCH::GRAYSKULL;
    //     }

    //     auto desc = backend::get_custom_device_desc(arch, mmio);
    //     d.emplace_back(arch, desc.soc_desc_yaml, desc.mmio, 0, context);
    // }
    // else
    // {
    //     int index = 0;
    //     for (auto desc : available_devices)
    //     {
    //         d.emplace_back(desc.arch, desc.soc_desc_yaml, desc.mmio, index++, context);
    //     }
    // }

    // if (d.empty())
    //     log_fatal(LogTTDevice, "No available devices detected (To run with golden device, set PYBUDA_DEVMODE=1)");

    // log_debug(LogTTDevice, "Available devices:");
    // for (int i = 0; i < (int)d.size(); ++i) log_debug(LogTTDevice, "  [{}] {} {}", i, d[i].type, d[i].arch);
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
