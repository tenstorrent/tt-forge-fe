// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "verif_ops.hpp"

#include <limits>
#include <tuple>

#include "utils/assert.hpp"

// Needed so the kernels can be ran parallelized.
#define INTRA_OP_PARALLEL

#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/Reduce.h>

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
    static_assert(
        op == ReduceOp::Max || op == ReduceOp::Sum || op == ReduceOp::LogicalOr, "Unsupported reduction operation");

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
    using reduce_result_t = typename decltype(std::function{do_reduce_op<op_result_t, reduce_op>})::result_type;

    static_assert(std::is_same_v<init_t, reduce_result_t>, "init_value type must match reduce_result_t");
    static_assert(
        std::is_same_v<
            typename decltype(std::function{reduce_vec_into_scalar<reduce_op, scalar_t, init_t>})::result_type,
            init_t>,
        "reduce_vec_into_scalar must return the same type as the final result - init_t");

    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int64_t vec_size = Vec::size();

    // The vectorized elementwise operation on inputs can only be performed if all inputs are contiguous, i.e.
    // for all inputs the stride is equal to 1.
    //
    // NOTE: this is the case because we cannot trivially load `vec_size` elements into a vectorized type (if they are
    // not contiguous in memory).
    bool vectorizable = false;

    for (int64_t input_id = 0; input_id < iter.ninputs(); ++input_id)
    {
        auto input = iter.input(input_id);
        vectorizable |= input.is_contiguous();
    }

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
        for (; i <= n - vec_size && vectorizable; i += vec_size)
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

bool is_supported_dtype(const torch::Tensor& a)
{
    switch (a.scalar_type())
    {
        case torch::kFloat:
        case torch::kDouble:
        case torch::kBFloat16:
        case torch::kHalf:
        case torch::kInt:
        case torch::kLong:
        case torch::kShort: return true;
        default: return false;
    }
}

bool has_special_values(const torch::Tensor& a)
{
    if (!at::is_floating_point(a))
    {
        // Only floating point types can have special values (NaN, Inf).
        return false;
    }

    auto options = a.options();
    options = options.dtype(torch::kBool);
    torch::Tensor output = at::empty({1}, options);

    auto a_flat = a.flatten();
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .check_all_same_dtype(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a_flat)
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

double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b)
{
    TT_ASSERT(
        a.dtype() == b.dtype(), "Input tensors must have the same data type, but got {} and {}", a.dtype(), b.dtype());
    TT_ASSERT(is_supported_dtype(a), "Unsupported data type for tensor a: {}", a.dtype());
    TT_ASSERT(is_supported_dtype(b), "Unsupported data type for tensor b: {}", b.dtype());

    torch::Tensor output = at::empty({1}, a.options());

    auto a_flat = a.flatten();
    auto b_flat = b.flatten();

    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a_flat)
                    .add_input(b_flat)
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

bool all_close(const torch::Tensor& a, const torch::Tensor& b, double rtol, double atol)
{
    TT_ASSERT(is_supported_dtype(a), "Unsupported data type for tensor a: {}", a.dtype());
    TT_ASSERT(is_supported_dtype(b), "Unsupported data type for tensor b: {}", b.dtype());
    TT_ASSERT(!has_special_values(a), "Tensor a contains NaN/Inf values");
    TT_ASSERT(!has_special_values(b), "Tensor b contains NaN/Inf values");

    auto a_flat = a.flatten();
    auto b_flat = b.flatten();

    if (!at::is_floating_point(a))
    {
        auto options = a.options().dtype(torch::kBool);
        torch::Tensor output = at::empty({1}, options);

        auto iter = at::TensorIteratorConfig()
                        .resize_outputs(false)
                        .declare_static_shape(output.sizes())
                        .check_all_same_dtype(false)
                        .add_output(output)
                        .add_input(a_flat)
                        .add_input(b_flat)
                        .build();

        // For non-floating point types, we need to check if the values are equal.
        AT_DISPATCH_INTEGRAL_TYPES(
            iter.input().scalar_type(),
            "all_close",
            [&]()
            {
                cpu_kernel_reduce_into_scalar<ReduceOp::LogicalOr>(
                    iter, [](scalar_t a_val, scalar_t b_val) -> bool { return a_val != b_val; }, false);
            });

        return !iter.output().item<bool>();
    }

    auto options = a.options();
    torch::Tensor output = at::empty({1}, options);

    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(output.sizes())
                    .add_output(output)
                    .add_input(a_flat)
                    .add_input(b_flat)
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

double calculate_mean(const torch::Tensor& tensor)
{
    TT_ASSERT(tensor.dim() == 1, "Input tensor must be 1D");
    if (torch::is_floating_point(tensor))
    {
        return tensor.mean().item<double>();
    }

    // If the tensor is not floating point, we need to convert it to double (for torch::mean to work).
    return tensor.mean(c10::kDouble).item<double>();
}

// Calculates the covariance between two tensor variables (1D tensors).
// Used as a building block for calculating the covariance matrix.
double cov_ij(
    const torch::Tensor& a,
    const torch::Tensor& b,
    std::optional<double> a_mean = std::nullopt,
    std::optional<double> b_mean = std::nullopt)
{
    TT_ASSERT(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TT_ASSERT(a.dim() == 1, "Input tensors must be 1D");

    if (!a_mean.has_value())
    {
        a_mean = calculate_mean(a);
    }

    if (!b_mean.has_value())
    {
        b_mean = calculate_mean(b);
    }

    auto output = at::empty({1}, a.options().dtype(torch::kDouble));
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(output.sizes())
                    .check_all_same_dtype(false)
                    .add_output(output)
                    .add_input(a)
                    .add_input(b)
                    .build();

    const auto N = a.numel();

    AT_DISPATCH_ALL_TYPES_AND2(
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

double calculate_tensor_pcc(const torch::Tensor& a, const torch::Tensor& b)
{
    TT_ASSERT(is_supported_dtype(a), "Unsupported data type for tensor a: {}", a.dtype());
    TT_ASSERT(is_supported_dtype(b), "Unsupported data type for tensor b: {}", b.dtype());
    auto a_flat = a.flatten();
    auto b_flat = b.flatten();

    TT_ASSERT(a_flat.numel() == b_flat.numel(), "Input tensors must have the same number of elements");

    double a_mean;
    double b_mean;

    a_mean = calculate_mean(a_flat);
    b_mean = calculate_mean(b_flat);

    // To calculate PCC we need to calculate the covariance matrix of the two tensors;
    // the two tensors are treated as two random variables (with N samples).
    //
    // The covariance matrix is square matrix (in this case 2x2) that describes the covariance between each pair of the
    // variables (tensors). Element (i, j) of the covariant matrix is calculated as follows, where x is the i-th
    // variable and y is the j-th variable:
    //
    // cov[i][j] = sum((x[k] - mean(x)) * (y[k] - mean(y))) / N, k goes from 0 to N-1
    //
    // PCC matrix also a square matrix and is calculated as follows (where x is the i-th variable and y is the j-th
    // variable):
    //
    // PCC[i][j] = cov[i][j] / (std(x) * std(y)), where std(x) is the standard deviation of x
    //
    // Standard deviation is in fact the square root of the variance, which is cov[i][i] (the covariance of the variable
    // with itself).
    //
    // So, to compute the PCC matrix we use the following formula:
    //
    // PCC[i][j] = cov[i][j] / (sqrt(cov[i][i]) * sqrt(cov[j][j]))
    //
    // Finally, the resulting PCC value is taken as the minimum element of the PCC matrix.

    auto options = a.options().dtype(torch::kDouble);
    auto cov = at::empty({4}, options);
    cov[0] = cov_ij(a_flat, a_flat, a_mean, a_mean);
    cov[1] = cov_ij(a_flat, b_flat, a_mean, b_mean);
    cov[2] = cov_ij(b_flat, a_flat, b_mean, a_mean);
    cov[3] = cov_ij(b_flat, b_flat, b_mean, b_mean);

    auto std_a = torch::sqrt(cov[0]);
    auto std_b = torch::sqrt(cov[3]);

    auto std_tensor = torch::stack({std_a * std_a, std_a * std_b, std_b * std_a, std_b * std_b}, 0 /* dim */);

    // There should be no NaN/Inf in the covariance matrix.
    TT_ASSERT(
        has_special_values(cov) == false, "Covariance matrix contains NaN/Inf values - possibly due to an overflow");

    auto pcc_tensor = torch::divide(cov, std_tensor);

    return pcc_tensor.min().item<double>();
}

}  // namespace tt
