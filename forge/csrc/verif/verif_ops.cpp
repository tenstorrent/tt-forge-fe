// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "verif_ops.hpp"

#include <limits>
#include <tuple>
#include <utils/assert.hpp>

#define INTRA_OP_PARALLEL

#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/Reduce.h>

#include <utils/logger.hpp>

namespace tt
{

enum class ReduceOp
{
    Max,
    Sum,
    LogicalOr,
};

// Performs the reduction operation `op` on the two scalar values `a` and `b`.
template <typename scalar_t, ReduceOp op>
inline auto do_reduce_op(scalar_t a, scalar_t b)
{
    if constexpr (op == ReduceOp::Max)
    {
        return std::max(a, b);
    }
    else if constexpr (op == ReduceOp::Sum)
    {
        return a + b;
    }
    else if constexpr (op == ReduceOp::LogicalOr)
    {
        return a || b;
    }
    else
    {
        TT_ASSERT(false, "Unsupported reduction operation");
    }
}

// Reduces the vectorized type `vec` into a scalar value using the reduction operation `op`.
template <ReduceOp op, typename scalar_t, typename init_t>
inline init_t reduce_vec_into_scalar(at::vec::Vectorized<scalar_t> vec, init_t init_value)
{
    auto ret = init_value;
    for (int64_t i = 0; i != vec.size(); i++)
    {
        ret = do_reduce_op<scalar_t, op>(ret, vec[i]);
    }
    return ret;
}

// Generic reduction operation for vectorized types. This function performs the reduction operation on the
// vectorized type `a` using the vectorized type `b` - in place.
template <ReduceOp op, typename scalar_t>
inline void vec_reduce_op(at::vec::Vectorized<scalar_t>& a, const at::vec::Vectorized<scalar_t>& b)
{
    if constexpr (op == ReduceOp::Max)
    {
        a = at::vec::maximum(a, b);
    }
    else if constexpr (op == ReduceOp::Sum)
    {
        a += b;
    }
    else if constexpr (op == ReduceOp::LogicalOr)
    {
        a = a || b;
    }
    else
    {
        TT_ASSERT(false, "Unsupported reduction operation");
    }
}

// Generic kernel template for performing elementwise nary operations on flattenened tensors and reducing the result
// (producing a single scalar value at the end). The whole operation (operation + reduction) is performed in-place
// (without allocating intermediate tensors).
//
// E.g. for checking if tensor contains special values (NaN, Inf) we can use this kernel to perform check on each
// element while reducing the result using the `logical OR` operation down to a single boolean value.
//
// Template parameters:
// - `reduce_op`: The reduction operation to be performed on the result of the elementwise operation.
// - `func_t`: The elementwise operation to be performed on the input tensors. This should be a callable object (e.g.
//    a lambda function) that takes N arguments (one for each input tensor) and returns a single value.
// - `init_t`: The type of the initial value for the reduction operation. This should be the same type as the result of
// the
//   reduction operation.
//
template <ReduceOp reduce_op, typename func_t, typename init_t>
inline void cpu_kernel_reduce_into_scalar(at::TensorIteratorBase& iter, func_t scalar_op, init_t init_value)
{
    using traits = function_traits<func_t>;
    // using scalar_t = typename traits::template arg<0>::type;
    using op_result_t = typename traits::result_type;
    using reduce_result_t = typename decltype(std::function{do_reduce_op<op_result_t, reduce_op>})::result_type;

    static_assert(std::is_same_v<reduce_result_t, op_result_t>, "scalar_op must return the same type as the result");
    static_assert(std::is_same_v<init_t, reduce_result_t>, "init_value type must match reduce_result_t");

    std::atomic<reduce_result_t> global_result = init_value;

    int64_t numel = iter.input().numel();
    auto loop_fn = [&](char** data, const int64_t* strides, int64_t n)
    {
        op_result_t chunk_result = init_value;

        int64_t i = 0;

        for (; i < n; ++i)
        {
            // Create the argument tuple for the elementwise operation and run it.
            // NOTE: starting from data[1], since data[0] is the output tensor.
            auto args = at::native::dereference<traits>(&data[1], &strides[1], i);
            auto result = std::apply(scalar_op, args);

            // Perform the reduction operation on the result to update the chunk result.
            chunk_result = do_reduce_op<op_result_t, reduce_op>(chunk_result, result);
        }

        // Update the global result with new chunk result atomically.
        auto expected = global_result.load();
        auto desired = do_reduce_op<reduce_result_t, reduce_op>(expected, chunk_result);
        while (!global_result.compare_exchange_weak(expected, desired))
        {
            desired = do_reduce_op<reduce_result_t, reduce_op>(expected, chunk_result);
        }
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

    iter.output().fill_(global_result.load());
}

// Generic kernel template for performing vectorized elementwise nary operations on flattenened tensors and reducing the
// result (producing a single scalar value at the end). The whole operation (operation + reduction) is performed
// in-place (without allocating intermediate tensors).
//
// E.g. for calculating max(|a - b|) we can use this kernel to perform |a - b| on each element while reducing the result
// using the `max` operation down to a single scalar value.
//
// Template parameters:
// - `reduce_op`: The reduction operation to be performed on the result of the elementwise operation.
// - `func_t`: The elementwise operation to be performed on the input tensors. This should be a callable object (e.g.
//   a lambda function) that takes N arguments (one for each input tensor) and returns a single value.
// - `vec_func_t`: The vectorized elementwise operation to be performed on the input tensors. This should be a callable
//   object (e.g. a lambda function) that takes N arguments (one for each input tensor) and returns a vectorized value.
// - `init_t`: The type of the initial value for the reduction operation. This should be the same type as the result of
//   the reduction operation.
//
template <ReduceOp reduce_op, typename func_t, typename vec_func_t, typename init_t>
inline void cpu_kernel_reduce_into_scalar_vec(
    at::TensorIteratorBase& iter, func_t scalar_op, vec_func_t vec_op, init_t init_val)
{
    using traits = function_traits<func_t>;
    using vec_traits = function_traits<vec_func_t>;
    using scalar_t = typename traits::template arg<0>::type;
    using op_result_t = typename traits::result_type;
    using vec_op_result_t = typename decltype(std::function{vec_op})::result_type;
    using reduce_result_t = typename decltype(std::function{do_reduce_op<op_result_t, reduce_op>})::result_type;

    static_assert(
        std::is_same_v<at::vec::Vectorized<op_result_t>, vec_op_result_t>,
        "vec_func_t must return at::vec::Vectorized<op_result_t>");
    static_assert(std::is_same_v<init_t, reduce_result_t>, "init_value type must match reduce_result_t");

    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int64_t vec_size = Vec::size();

    std::atomic<reduce_result_t> global_result = init_val;

    int64_t numel = iter.input().numel();
    auto loop_fn = [&](char** data, const int64_t* strides, int64_t n)
    {
        reduce_result_t chunk_result = init_val;

        Vec chunk_result_vec = Vec(init_val);

        // This is a dummy value to be used for dereference_vec() - the function has an option to use a default scalar
        // value in case one of the operands is a scalar.
        Vec opt_scalar = Vec(init_val);
        // This is a dummy value to be used for dereference_vec() - inidcates that we don't have any arguments that are
        // scalars.
        constexpr int64_t opt_scalar_index = 0;

        int64_t i = 0;
        for (; i <= n - vec_size; i += vec_size)
        {
            // Load the data for the current chunk and perform the vectorized operation.
            // NOTE: starting from data[1], since data[0] is the output tensor.
            auto args = at::native::dereference_vec<vec_traits>(&data[1], opt_scalar, opt_scalar_index, i);
            auto out = std::apply(vec_op, std::move(args));

            vec_reduce_op<reduce_op>(chunk_result_vec, std::move(out));
        }

        chunk_result = reduce_vec_into_scalar<reduce_op>(std::move(chunk_result_vec), init_val);

        for (; i < n; ++i)
        {
            // Create the argument tuple for the elementwise operation and run it.
            // NOTE: starting from data[1], since data[0] is the output tensor.
            auto args = at::native::dereference<traits>(&data[1], &strides[1], i);
            auto result = std::apply(scalar_op, args);
            chunk_result = do_reduce_op<reduce_result_t, reduce_op>(chunk_result, result);
        }

        // Update the global result atomically.
        auto expected = global_result.load();
        auto desired = do_reduce_op<reduce_result_t, reduce_op>(expected, chunk_result);
        while (!global_result.compare_exchange_weak(expected, desired))
        {
            desired = do_reduce_op<reduce_result_t, reduce_op>(expected, chunk_result);
        }
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

    iter.output().fill_(global_result.load());
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

    AT_DISPATCH_ALL_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iter.input().scalar_type(),
        "has_special_values",
        [&]()
        {
            cpu_kernel_reduce_into_scalar<ReduceOp::LogicalOr>(
                iter, [](scalar_t val) -> bool { return std::isinf(val) || std::isnan(val); }, false);
        });

    return iter.output().item<bool>();
}

double max_abs_diff(torch::Tensor& a, torch::Tensor& b)
{
    TT_ASSERT(
        a.dtype() == b.dtype(), "Input tensors must have the same data type, but got {} and {}", a.dtype(), b.dtype());
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

    AT_DISPATCH_ALL_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iter.dtype(),
        "max_abs_diff",
        [&]()
        {
            cpu_kernel_reduce_into_scalar_vec<ReduceOp::Max>(
                iter,
                [](scalar_t a_val, scalar_t b_val) -> scalar_t
                { return static_cast<scalar_t>(std::abs(a_val - b_val)); },
                [](at::native::Vectorized<scalar_t> a,
                   at::native::Vectorized<scalar_t> b) -> at::native::Vectorized<scalar_t>
                {
                    auto delta = (a - b).abs();
                    return delta;
                },
                std::numeric_limits<scalar_t>::lowest());
        });

    return iter.output().item<double>();
}

bool unsupported_dtypes(torch::Tensor& a)
{
    switch (a.scalar_type())
    {
        case torch::kFloat:
        case torch::kDouble:
        case torch::kBFloat16:
        case torch::kHalf: return false;
        default: return true;
    }
}

bool all_close(torch::Tensor a, torch::Tensor b, double rtol, double atol, bool equal_nan)
{
    auto options = a.options();
    torch::Tensor output = at::empty({1}, options);

    if (unsupported_dtypes(a) || unsupported_dtypes(b) || has_special_values(a) || has_special_values(b))
    {
        return at::allclose(a, b, rtol, atol, equal_nan);
    }

    a = a.flatten();
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a)
                    .add_input(b)
                    .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iter.input().scalar_type(),
        "all_close",
        [&]()
        {
            const auto atol_vec = at::native::Vectorized<scalar_t>(atol);
            const auto rtol_vec = at::native::Vectorized<scalar_t>(rtol);

            cpu_kernel_reduce_into_scalar_vec<ReduceOp::Max>(
                iter,
                [rtol, atol](scalar_t a_val, scalar_t b_val) -> scalar_t
                { return std::abs(a_val - b_val) - atol - rtol * std::abs(b_val); },
                [&rtol_vec, &atol_vec](at::native::Vectorized<scalar_t> a, at::native::Vectorized<scalar_t> b)
                    -> at::native::Vectorized<scalar_t>
                {
                    auto delta = (a - b).abs() - atol_vec - b.abs() * rtol_vec;
                    return delta;
                },
                std::numeric_limits<scalar_t>::lowest());
        });

    return iter.output().item<double>() <= 0.0;
}

double cov_ij(
    torch::Tensor& a,
    torch::Tensor& b,
    std::optional<double> a_mean = std::nullopt,
    std::optional<double> b_mean = std::nullopt)
{
    TT_ASSERT(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TT_ASSERT(a.dim() == 1, "Input tensors must be 1D");
    TT_ASSERT(b.dim() == 1, "Input tensors must be 1D");
    TT_ASSERT(a.strides() == b.strides(), "Input tensors must have the same strides");
    TT_ASSERT(a.is_contiguous() && b.is_contiguous(), "Input tensors must be contiguous");

    if (!a_mean.has_value())
    {
        a_mean = a.mean().item<double>();
    }

    if (!b_mean.has_value())
    {
        b_mean = b.mean().item<double>();
    }

    auto output = at::empty({1}, a.options());
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a)
                    .add_input(b)
                    .build();

    const auto N = a.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iter.input().scalar_type(),
        "cov_ij",
        [&]()
        {
            cpu_kernel_reduce_into_scalar<ReduceOp::Sum>(
                iter,
                [&a_mean, &b_mean, &N](scalar_t a_val, scalar_t b_val) -> double
                {
                    auto d_a = static_cast<double>(a_val) - a_mean.value();
                    auto d_b = static_cast<double>(b_val) - b_mean.value();
                    return d_a / N * d_b;
                },
                static_cast<double>(0));
        });

    return iter.output().item<double>();
}

double calculate_tensor_pcc(torch::Tensor& a, torch::Tensor& b)
{
    auto a_flat = a.flatten();
    auto b_flat = b.flatten();

    if (!a_flat.is_contiguous())
    {
        a_flat = a_flat.contiguous();
    }

    if (!b_flat.is_contiguous())
    {
        b_flat = b_flat.contiguous();
    }

    auto options = a.options();
    torch::Tensor output = at::empty({1}, options);

    TT_ASSERT(a_flat.numel() == b_flat.numel(), "Input tensors must have the same number of elements");

    if (unsupported_dtypes(a_flat) || unsupported_dtypes(b_flat))
    {
        // Fallback to torch impl.
        return at::min(at::corrcoef(at::stack({a_flat, b_flat}))).item<double>();
    }

    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a_flat)
                    .add_input(b_flat)
                    .build();

    auto a_mean = a_flat.mean().item<double>();
    auto b_mean = b_flat.mean().item<double>();

    auto cov = at::empty({4}, options);
    cov[0] = cov_ij(a_flat, a_flat, a_mean, a_mean);
    cov[1] = cov_ij(a_flat, b_flat, a_mean, b_mean);
    cov[2] = cov_ij(b_flat, a_flat, b_mean, a_mean);
    cov[3] = cov_ij(b_flat, b_flat, b_mean, b_mean);

    auto std_a = std::sqrt(cov[0].item<double>());
    auto std_b = std::sqrt(cov[3].item<double>());

    double min = std::numeric_limits<double>::max();

    // There should be no NaN/Inf in the covariance matrix.
    TT_ASSERT(
        has_special_values(cov) == false, "Covariance matrix contains NaN/Inf values - possibly due to an overflow");

    min = std::min(min, cov[0].item<double>() / (std_a * std_a));
    min = std::min(min, cov[1].item<double>() / (std_a * std_b));
    min = std::min(min, cov[2].item<double>() / (std_b * std_a));
    min = std::min(min, cov[3].item<double>() / (std_b * std_b));

    return min;
}

}  // namespace tt
