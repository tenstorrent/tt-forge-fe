// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <variant>

#include "torch/torch.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "utils/assert.hpp"

namespace tt
{

target::DataType torch_scalar_type_to_dt(torch::ScalarType st);

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

class Tensor
{
   public:
    Tensor(torch::Tensor& tensor) : tensor_storage(tensor), rt_tensor(std::nullopt)
    {
        auto data = std::shared_ptr<void>(tensor.data_ptr(), [](void*) {});

        auto shape = std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
        auto stride = std::vector<uint32_t>(tensor.strides().begin(), tensor.strides().end());

        rt_tensor = runtime::createTensor(
            data, shape, stride, tensor.element_size(), torch_scalar_type_to_dt(tensor.scalar_type()));
    }

    Tensor(runtime::Tensor& tensor) : tensor_storage(tensor), rt_tensor(tensor) {}

    std::shared_ptr<void> storage() const { return tensor_storage.borrow_data(); }

    runtime::Tensor get_runtime_tensor()
    {
        TT_ASSERT(rt_tensor.has_value());
        return *rt_tensor;
    }

   private:
    TensorStorage tensor_storage;
    std::optional<runtime::Tensor> rt_tensor;
};

class TensorPool
{
   public:
    TensorPool() = default;

    void insert(std::string name, torch::Tensor& tensor)
    {
        auto t = Tensor(tensor);
        insert(name, t);
    }

    Tensor get_tensor(std::string name) const
    {
        TT_ASSERT(tensor_name_to_idx.find(name) != tensor_name_to_idx.end(), "Tensor {} not found", name);
        size_t idx = tensor_name_to_idx.at(name);
        return tensors[idx];
    }

   private:
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, size_t> tensor_name_to_idx;

    void insert(std::string name, Tensor& tensor)
    {
        if (tensor_name_to_idx.find(name) != tensor_name_to_idx.end())
        {
            TT_ASSERT(
                tensor.storage() == tensors[tensor_name_to_idx[name]].storage(),
                "Different tensor with the same name ({}) already exists",
                name);
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
        case target::DataType::Float16: return torch::ScalarType::Half;
        case target::DataType::Float32: return torch::ScalarType::Float;
        case target::DataType::BFloat16: return torch::ScalarType::BFloat16;
        default: TT_THROW(false, "Unhandled dtype {}", target::EnumNameDataType(df));
    }

    __builtin_unreachable();
}

}  // namespace tt
