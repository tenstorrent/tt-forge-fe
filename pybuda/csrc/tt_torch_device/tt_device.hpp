// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

/*
 * Thin API layer between torch_device_impl.cpp (torch interop) and tt_device.cpp (backend interop)
 */

#include <torch/python.h>
#include <torch/torch.h>

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include "utils/assert.hpp"
#include "utils/env.hpp"
#include "utils/logger.hpp"
#include "pybuda/csrc/backend_api/arch_type.hpp"

namespace tt
{
constexpr inline c10::DeviceType TT = c10::DeviceType::PrivateUse1;
constexpr inline int kTileDim = 32;


struct TTMetaData : public c10::BackendMeta {
    torch::IntArrayRef original_shape;
    bool runtime_transformed = false;
    bool created_on_device = false;
    int unique_output_id = -1;
};

struct PyBudaTensorDesc
{
    std::string name;
    std::vector<std::int64_t> shape;
    int ptr = -1;
    std::optional<torch::Tensor> constant;

    PyBudaTensorDesc(
        std::string name,
        std::vector<std::int64_t> shape,
        int ptr,
        std::optional<torch::Tensor> constant) :
        name(name), shape(shape), ptr(ptr), constant(constant)
    {
    }
};


struct Program
{
    std::string name;
    std::map<std::string, std::string> parameters;

    Program(std::string const& name, std::map<std::string, std::string> const& parameters) :
        name(name), parameters(parameters)
    {
    }
};

struct CompileRequest
{
    std::string netlist_path;
    std::string output_dir;
    std::map<int, std::vector<PyBudaTensorDesc>> inputs;  // one vector per program
    std::map<int, std::vector<std::string>> input_runtime_transforms;
    std::map<int, std::vector<std::vector<int>>> input_tile_bcast_dims;
    std::vector<PyBudaTensorDesc> constants;
    std::vector<PyBudaTensorDesc> parameters;
    std::map<int, std::vector<PyBudaTensorDesc>> outputs; 
    std::map<int, std::vector<std::string>> output_runtime_transforms;

    CompileRequest(
        std::string const& netlist_path,
        std::string output_dir,
        std::map<int, std::vector<PyBudaTensorDesc>> const& inputs,
        std::map<int, std::vector<std::string>> const& input_runtime_transforms,
        std::map<int, std::vector<std::vector<int>>> const& input_tile_bcast_dims,
        std::vector<PyBudaTensorDesc> const& constants,
        std::vector<PyBudaTensorDesc> const& parameters,
        std::map<int, std::vector<PyBudaTensorDesc>> const& outputs,
        std::map<int, std::vector<std::string>> const& output_runtime_transforms) :
        netlist_path(netlist_path),
        output_dir(output_dir),
        inputs(inputs),
        input_runtime_transforms(input_runtime_transforms),
        input_tile_bcast_dims(input_tile_bcast_dims),
        constants(constants),
        parameters(parameters),
        outputs(outputs),
        output_runtime_transforms(output_runtime_transforms)
    {
    }
};

struct Workload
{
    std::string output_dir;
    std::map<int, std::vector<PyBudaTensorDesc>> inputs;
    std::vector<PyBudaTensorDesc> constants;
    std::vector<PyBudaTensorDesc> parameters;
    std::map<int, std::vector<PyBudaTensorDesc>> outputs;
    bool initialized = false;
    std::unordered_map<int, bool> subgraph_link_tensor_populated;

    Workload(
        std::string output_dir,
        std::map<int, std::vector<PyBudaTensorDesc>> const& inputs, // a vector per program
        std::vector<PyBudaTensorDesc> const& constants,
        std::vector<PyBudaTensorDesc> const& parameters,
        std::map<int, std::vector<PyBudaTensorDesc>> const& outputs) :
        output_dir(output_dir),
        inputs(inputs),
        constants(constants),
        parameters(parameters),
        outputs(outputs)
    {
    }
};

struct TTContext
{
    std::atomic_bool initialized = false;
    ~TTContext();
};

using Fence = std::uint64_t;
using ResourceID = std::uint64_t;

// 1to1 mapping of physical devices plugged into this machine and TTDevice
struct TTDevice
{
    ARCH arch;
    std::string soc_desc_yaml;
    bool mmio;
    int index;
    std::shared_ptr<TTContext> context;
    std::map<int, std::vector<std::string>> input_runtime_transforms;
    std::map<int, std::vector<std::vector<int>>> input_tile_bcast_dims;
    std::map<int, std::vector<std::string>> output_runtime_transforms;
    bool initialized = false;
    std::unordered_map<int, std::vector<int>> subgraph_to_tensor_uid_on_device;

    TTDevice(
        ARCH arch, std::string soc_desc_yaml, bool mmio, int index, std::shared_ptr<TTContext> context) :
        arch(arch), soc_desc_yaml(soc_desc_yaml), mmio(mmio), index(index), context(context)
    {
    }
};

using FreePytorchTensorDescFn = void(void*);
void register_output_runtime_transform(torch::Tensor const& tensor, std::string transform);
void register__ordered_input_runtime_transforms(std::vector<std::string> input_transforms);
std::string get_runtime_transform(torch::Tensor const& tensor);
std::vector<TTDevice> query_available_tt_devices();
const TTDevice& get_default_tt_device();
std::vector<TTDevice> get_available_tt_devices();
std::string device_type_name(c10::DeviceType type, bool lower_case = false);
torch::Device torch_device_at_index(std::int64_t index);

std::vector<const void*> get_copied_inputs();
std::shared_ptr<Workload> compile(
    TTDevice& device, CompileRequest const& compile_request);
void push_tensor(
    torch::Tensor & tensor,
    std::string const& info = "",
    std::optional<int> ptr = std::nullopt);

std::vector<torch::Tensor> dispatch(
    TTDevice & device,
    std::shared_ptr<Workload> workload,
    std::vector<Program> const& programs,
    std::vector<torch::Tensor> & inputs,
    int subgraph_idx,
    bool const & is_compile);
std::string get_device_cluster_yaml(TTDevice const&);
std::string to_string(TTDevice const& d);
torch::Device torch_device(TTDevice const& d);

torch::Tensor eval_runtime_transform(const torch::Tensor& tensor, std::string transform, std::vector<int> &tile_bcast_dims);
bool is_created_on_device(const torch::Tensor& tensor);
int unique_id(const torch::Tensor& tensor);
torch::Tensor narrow_to_pytorch(const torch::Tensor& tensor, torch::IntArrayRef original_shape);
std::vector<size_t> original_shape(const torch::Tensor& tensor);

template <typename T>
inline T align_up_tile(T d)
{
    d -= 1;
    return static_cast<T>(d - (d % kTileDim) + kTileDim);
}
}  // namespace tt
