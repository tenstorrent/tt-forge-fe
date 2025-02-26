// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <variant>

#include "runtime/tt_device.hpp"
#include "torch/torch.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "utils/assert.hpp"

namespace tt
{

target::DataType torch_scalar_type_to_dt(torch::ScalarType st);
torch::ScalarType dt_to_torch_scalar_type(target::DataType df);

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

class TensorStorage
{
   public:
    explicit TensorStorage(torch::Tensor& tensor) : storage(tensor) {}
    explicit TensorStorage(runtime::Tensor& tensor) : storage(tensor) {}

    std::shared_ptr<void> borrow_data() const
    {
        return std::visit(
            [](auto&& arg) -> std::shared_ptr<void>
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, torch::Tensor>)
                {
                    return std::shared_ptr<void>(arg.data_ptr(), [arg](void*) { (void)arg; });
                }
                else if constexpr (std::is_same_v<T, runtime::Tensor>)
                {
                    return arg.data;
                }
            },
            storage);
    }

   private:
    std::variant<torch::Tensor, runtime::Tensor> storage;
};

class TensorImpl : public std::enable_shared_from_this<TensorImpl>
{
   public:
    TensorImpl(torch::Tensor& tensor) : tensor_storage(tensor), rt_tensor(std::nullopt)
    {
        auto shape = std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
        auto stride = std::vector<uint32_t>(tensor.strides().begin(), tensor.strides().end());

        desc.shape = shape;
        desc.stride = stride;
        desc.itemsize = tensor.element_size();
        desc.dataType = torch_scalar_type_to_dt(tensor.scalar_type());
    }

    TensorImpl(runtime::Tensor& tensor, runtime::TensorDesc tensor_desc) :
        tensor_storage(tensor), desc(tensor_desc), rt_tensor(tensor)
    {
    }

    std::shared_ptr<void> storage() const { return tensor_storage.borrow_data(); }

    runtime::Tensor& get_runtime_tensor()
    {
        TT_ASSERT(rt_tensor.has_value());
        return *rt_tensor;
    }

    void set_runtime_tensor(runtime::Tensor tensor) { rt_tensor.value() = tensor; }

    torch::Tensor to_host() const
    {
        TT_ASSERT(rt_tensor.has_value());

        constexpr bool untilize_tensor = true;
        auto host = tt::runtime::toHost(rt_tensor.value(), untilize_tensor);
        auto torch_tensor = at::empty_strided(
            as_vec_int64(desc.shape), as_vec_int64(desc.stride), dt_to_torch_scalar_type(desc.dataType));

        tt::runtime::memcpy(torch_tensor.data_ptr(), host);

        return torch_tensor;
    }

    void to_device(const size_t device_id, runtime::Layout& layout)
    {
        TT_ASSERT(!rt_tensor.has_value());
        auto device = TTSystem::get_system().devices[device_id];

        rt_tensor =
            runtime::createTensor(tensor_storage.borrow_data(), desc.shape, desc.stride, desc.itemsize, desc.dataType);
        rt_tensor = tt::runtime::toLayout(rt_tensor.value(), *device->rt_device, layout);
    }

    void update_host_data()
    {
        TT_ASSERT(rt_tensor.has_value());
        constexpr bool untilize_tensor = true;
        auto host = tt::runtime::toHost(rt_tensor.value(), untilize_tensor);

        TT_ASSERT(storage().get() != nullptr);
        tt::runtime::memcpy(storage().get(), host);
    }

    bool on_device() const { return rt_tensor.has_value(); }

    void detach_from_device() { rt_tensor.reset(); }

    runtime::TensorDesc tensor_desc() const { return desc; }

   private:
    TensorStorage tensor_storage;
    runtime::TensorDesc desc;
    std::optional<runtime::Tensor> rt_tensor;
};

class Tensor
{
   public:
    Tensor(torch::Tensor& tensor) : impl(new TensorImpl(tensor)) {}
    Tensor(runtime::Tensor& tensor, runtime::TensorDesc tensor_desc) : impl(new TensorImpl(tensor, tensor_desc)) {}

    std::shared_ptr<void> storage() const { return impl->storage(); }

    runtime::Tensor& get_runtime_tensor() { return impl->get_runtime_tensor(); }

    void set_runtime_tensor(runtime::Tensor tensor) { impl->set_runtime_tensor(tensor); }

    torch::Tensor to_host() const { return impl->to_host(); }

    void to_device(const size_t device_id, runtime::Layout& layout) { impl->to_device(device_id, layout); }

    void update_host_data() { impl->update_host_data(); }

    bool on_device() const { return impl->on_device(); }

    runtime::TensorDesc tensor_desc() const { return impl->tensor_desc(); }

    void detach_from_device() { impl->detach_from_device(); }

   private:
    std::shared_ptr<TensorImpl> impl;
};

// Class containing all (persistent) tensors across all programs.
class TensorPool
{
   public:
    TensorPool() = default;

    void insert(const std::string& name, torch::Tensor& tensor)
    {
        auto t = Tensor(tensor);
        insert(name, t);
    }

    Tensor get_tensor(const std::string& name) const
    {
        TT_ASSERT(tensor_name_to_idx.find(name) != tensor_name_to_idx.end(), "Tensor {} not found", name);
        size_t idx = tensor_name_to_idx.at(name);
        return tensors[idx];
    }

    bool exists(const std::string& name) const { return tensor_name_to_idx.find(name) != tensor_name_to_idx.end(); }

    void update_tensor(const std::string& name, tt::Tensor& tensor)
    {
        TT_ASSERT(tensor_name_to_idx.find(name) != tensor_name_to_idx.end(), "Tensor {} not found", name);
        size_t idx = tensor_name_to_idx.at(name);
        tensors.at(idx).get_runtime_tensor() = tensor.get_runtime_tensor();
    }

   private:
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, size_t> tensor_name_to_idx;

    void insert(std::string name, Tensor& tensor)
    {
        if (tensor_name_to_idx.find(name) != tensor_name_to_idx.end())
        {
            // TT_ASSERT(
            //     tensor.storage() == tensors[tensor_name_to_idx[name]].storage(),
            //     "Different tensor with the same name ({}) already exists",
            //     name);
            return;
        }

        tensor_name_to_idx[name] = tensors.size();
        tensors.push_back(tensor);
    }
};

inline target::DataType torch_scalar_type_to_dt(torch::ScalarType st)
{
    switch (st)
    {
        case torch::ScalarType::Byte: return target::DataType::UInt8;
        case torch::ScalarType::Char: return target::DataType::UInt8;
        case torch::ScalarType::Short: return target::DataType::UInt16;
        case torch::ScalarType::Int: return target::DataType::Int32;
        case torch::ScalarType::Long: return target::DataType::UInt32;
        case torch::ScalarType::Half: return target::DataType::Float16;
        case torch::ScalarType::Float: return target::DataType::Float32;
        // case torch::ScalarType::Double:
        // case torch::ScalarType::ComplexHalf:
        // case torch::ScalarType::ComplexFloat:
        // case torch::ScalarType::ComplexDouble:
        // case torch::ScalarType::Bool:
        case torch::ScalarType::BFloat16: return target::DataType::BFloat16;
        default: TT_THROW(false, "Unhandled scalar type {}", st);
    }

    __builtin_unreachable();
}

inline torch::ScalarType dt_to_torch_scalar_type(target::DataType df)
{
    switch (df)
    {
        case target::DataType::UInt8: return torch::ScalarType::Byte;
        case target::DataType::UInt16: return torch::ScalarType::Short;
        case target::DataType::UInt32: return torch::ScalarType::Int;
        case target::DataType::Int32: return torch::ScalarType::Int;
        case target::DataType::Float16: return torch::ScalarType::Half;
        case target::DataType::Float32: return torch::ScalarType::Float;
        case target::DataType::BFloat16: return torch::ScalarType::BFloat16;
        default: TT_THROW(false, "Unhandled dtype {}", target::EnumNameDataType(df));
    }

    __builtin_unreachable();
}

}  // namespace tt
