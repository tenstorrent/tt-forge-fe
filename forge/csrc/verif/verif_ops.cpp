// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "verif_ops.hpp"

#include <utils/assert.hpp>

#define INTRA_OP_PARALLEL

#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/Reduce.h>

#include <limits>
#include <utils/logger.hpp>

namespace tt
{

torch::Tensor is_close(torch::Tensor a, torch::Tensor b, double rtol, double atol, bool equal_nan)
{
    return torch::empty({1});
}
torch::Tensor all_close(torch::Tensor a, torch::Tensor b, double rtol, double atol, bool equal_nan)
{
    return torch::empty({1});
}

template <typename T>
T reduce_max(at::vec::Vectorized<T> vec)
{
    T ret = 0;
    for (int64_t i = 0; i != vec.size(); i++)
    {
        ret = std::max(ret, vec[i]);
    }
    return ret;
}

enum class ReduceOp
{
    Max,
    Min,
    LogicalOr,
    LogicalAnd,
};

template <typename scalar_t, ReduceOp op>
inline auto do_reduce_op(scalar_t a, scalar_t b)
{
    if constexpr (op == ReduceOp::Max)
    {
        return std::max(a, b);
    }
    else if constexpr (op == ReduceOp::Min)
    {
        return std::min(a, b);
    }
    else if constexpr (op == ReduceOp::LogicalOr)
    {
        return a || b;
    }
    else if constexpr (op == ReduceOp::LogicalAnd)
    {
        return a && b;
    }
    else
    {
        TT_ASSERT(false, "Unsupported reduction operation");
    }
}

template <ReduceOp op, typename scalar_t>
inline scalar_t scalar_reduce_op(at::vec::Vectorized<scalar_t> vec)
{
    scalar_t ret = vec[0];
    for (int64_t i = 1; i != vec.size(); i++)
    {
        ret = do_reduce_op<scalar_t, op>(ret, vec[i]);
    }
    return ret;
}

template <ReduceOp op, typename scalar_t>
inline void vec_reduce_op(at::vec::Vectorized<scalar_t>& a, const at::vec::Vectorized<scalar_t>& b)
{
    if constexpr (op == ReduceOp::Max)
    {
        a = at::vec::maximum(a, b);
    }
    else if constexpr (op == ReduceOp::Min)
    {
        a = at::vec::minimum(a, b);
    }
    else if constexpr (op == ReduceOp::LogicalOr)
    {
        a = a || b;
    }
    else if constexpr (op == ReduceOp::LogicalAnd)
    {
        a = a && b;
    }
    else
    {
        TT_ASSERT(false, "Unsupported reduction operation");
    }
}

template <ReduceOp reduce_op, typename func_t>
inline void cpu_kernel_reduce_into_scalar(at::TensorIteratorBase& iter, func_t scalar_op)
{
    using traits = unary_function_traits<func_t>;
    using scalar_t = typename traits::arg1_t;
    using result_t = decltype(scalar_op(typename traits::arg1_t{}));

    std::atomic<bool> barrier = std::numeric_limits<scalar_t>::min();
    // FIXME: Need to pass in the init value
    std::atomic<result_t> result = false;  // scalar_op(scalar_t{});

    int64_t numel = iter.input().numel();
    auto loop_fn = [&](char** data, const int64_t* strides, int64_t n)
    {
        auto first_val = *reinterpret_cast<scalar_t*>(data[1]);
        result_t chunk_result = scalar_op(first_val);

        // FIXME: Handle different number of input tensors
        auto* a_data = data[1];
        int64_t i = 1;

        for (; i < n; ++i)
        {
            auto a_val = *reinterpret_cast<scalar_t*>(a_data + i * sizeof(scalar_t));
            chunk_result = do_reduce_op<result_t, reduce_op>(chunk_result, scalar_op(a_val));
        }

        // Wait for atomic to go false
        bool expected = false;
        bool desired = true;
        while (barrier.compare_exchange_weak(expected, desired))
        {
            expected = false;
        }

        result = do_reduce_op<result_t, reduce_op>(result, chunk_result);

        barrier = false;
    };

    int64_t grain_size = at::internal::GRAIN_SIZE;
    if (numel < grain_size || at::get_num_threads() == 1)
    {
        iter.serial_for_each(loop_fn, {0, numel});
    }
    else
    {
        at::parallel_for(
            0, numel, grain_size, [&](int64_t begin, int64_t end) { iter.serial_for_each(loop_fn, {begin, end}); });
    }

    iter.output().fill_(result.load());
}

template <ReduceOp reduce_op, typename func_t, typename vec_func_t>
inline void cpu_kernel_reduce_into_scalar_vec(at::TensorIteratorBase& iter, func_t scalar_op, vec_func_t vec_op)
{
    using traits = binary_function_traits<func_t>;
    using scalar_t = typename traits::arg1_t;

    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int64_t vec_size = Vec::size();

    std::atomic<bool> barrier = std::numeric_limits<scalar_t>::min();
    std::atomic<scalar_t> result = std::numeric_limits<scalar_t>::min();

    int64_t numel = iter.input().numel();
    auto loop_fn = [&](char** data, const int64_t* strides, int64_t n)
    {
        scalar_t chunk_result = std::numeric_limits<scalar_t>::min();
        auto* a_data = data[1];
        auto* b_data = data[2];

        int64_t i = 0;

        Vec max_vec = Vec(std::numeric_limits<scalar_t>::min());

        for (; i <= n - 2 * vec_size; i += 2 * vec_size)
        {
            auto a_vec = Vec::loadu(a_data + i * sizeof(scalar_t));
            auto b_vec = Vec::loadu(b_data + i * sizeof(scalar_t));
            Vec out = vec_op(a_vec, b_vec);

            vec_reduce_op<reduce_op>(max_vec, out);

            auto a_vec2 = Vec::loadu(a_data + (i + vec_size) * sizeof(scalar_t));
            auto b_vec2 = Vec::loadu(b_data + (i + vec_size) * sizeof(scalar_t));
            out = vec_op(a_vec2, b_vec2);

            vec_reduce_op<reduce_op>(max_vec, out);
        }

        chunk_result = scalar_reduce_op<reduce_op>(max_vec);

        for (; i < n; ++i)
        {
            auto a_val = *reinterpret_cast<scalar_t*>(a_data + i * sizeof(scalar_t));
            auto b_val = *reinterpret_cast<scalar_t*>(b_data + i * sizeof(scalar_t));
            auto out = scalar_op(a_val, b_val);
            if (chunk_result < out)
            {
                chunk_result = out;
            }
        }

        // Wait for atomic to go false
        bool expected = false;
        bool desired = true;
        while (barrier.compare_exchange_weak(expected, desired))
        {
            expected = false;
        }

        result = scalar_op(result, chunk_result);

        barrier = false;
    };

    int64_t grain_size = at::internal::GRAIN_SIZE;
    if (numel < grain_size || at::get_num_threads() == 1)
    {
        iter.serial_for_each(loop_fn, {0, numel});
    }
    else
    {
        at::parallel_for(
            0, numel, grain_size, [&](int64_t begin, int64_t end) { iter.serial_for_each(loop_fn, {begin, end}); });
    }

    iter.output().fill_(result.load());
}

bool has_special_values(torch::Tensor& a)
{
    auto options = a.options();
    options = options.dtype(torch::kBool);
    torch::Tensor output = at::empty({1}, options);

    a = a.flatten();
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .check_all_same_dtype(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a)
                    .build();

    AT_DISPATCH_FLOATING_TYPES(
        iter.input().scalar_type(),
        "has_special_values",
        [&]()
        {
            cpu_kernel_reduce_into_scalar<ReduceOp::LogicalOr>(
                iter, [](scalar_t val) -> bool { return std::isinf(val) || std::isnan(val); });
        });

    return iter.output().item<bool>();
}

double max_abs_diff(torch::Tensor& a, torch::Tensor& b)
{
    torch::Tensor output = at::empty({1}, a.options());

    a = a.flatten();
    b = b.flatten();
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a)
                    .add_input(b)
                    .build();

    AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(),
        "max_abs_diff",
        [&]()
        {
            cpu_kernel_reduce_into_scalar_vec<ReduceOp::Max>(
                iter,
                [](scalar_t a_val, scalar_t b_val) -> double
                { return std::max(static_cast<scalar_t>(std::abs(a_val - b_val)), static_cast<scalar_t>(0)); },
                [](at::native::Vectorized<scalar_t> a,
                   at::native::Vectorized<scalar_t> b) -> at::native::Vectorized<scalar_t>
                {
                    auto delta = (a - b).abs();
                    return delta;
                });
        });

    return iter.output().item<double>();
}

}  // namespace tt
