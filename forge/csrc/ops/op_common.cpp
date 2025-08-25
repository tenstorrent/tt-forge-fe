// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op_common.hpp"

#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_forge/common.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"
#include "utils/assert.hpp"

namespace tt
{
namespace ops
{
namespace op_common
{
std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> eltwise_nary_shape(
    const std::vector<std::vector<uint32_t>> &in_shapes)
{
    std::vector<graphlib::DimBroadcast> broadcast;
    size_t max_dims = 0;
    for (const auto &shape : in_shapes) max_dims = std::max(max_dims, shape.size());

    std::vector<std::vector<uint32_t>> padded_shapes = in_shapes;
    for (auto &shape : padded_shapes)
        while (shape.size() < max_dims) shape.insert(shape.begin(), 1);

    std::vector<uint32_t> output_shape(max_dims);
    for (size_t dim = 0; dim < max_dims; ++dim)
    {
        uint32_t max_size = 1;
        for (const auto &shape : padded_shapes) max_size = std::max(max_size, shape[dim]);

        output_shape[dim] = max_size;

        for (size_t op_idx = 0; op_idx < padded_shapes.size(); ++op_idx)
        {
            if (padded_shapes[op_idx][dim] == max_size)
                continue;

            TT_ASSERT(
                padded_shapes[op_idx][dim] == 1,
                "Eltwise ops must have same shape or operand must be 1 wide to broadcast");

            broadcast.push_back(
                {static_cast<int>(op_idx), static_cast<int>(dim) - static_cast<int>(max_dims), max_size});
        }
    }

    return {graphlib::Shape::create(output_shape), broadcast};
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> compute_elementwise_binary_shape(
    const std::vector<std::vector<uint32_t>> &in_shapes)
{
    TT_ASSERT(in_shapes.size() == 2, "Elementwise binary ops should have exactly two input shapes.");

    std::vector<graphlib::DimBroadcast> broadcast;
    std::vector<uint32_t> output_shape;

    std::vector<uint32_t> shape0 = in_shapes[0];
    std::vector<uint32_t> shape1 = in_shapes[1];

    // Add leading 1s to the shorter shape
    while (shape0.size() < shape1.size())
    {
        shape0.insert(shape0.begin(), 1);
    }

    while (shape1.size() < shape0.size())
    {
        shape1.insert(shape1.begin(), 1);
    }

    output_shape.resize(shape0.size());

    for (size_t dim = 0; dim < shape0.size(); dim++)
    {
        if (shape0[dim] == shape1[dim])
        {
            output_shape[dim] = shape0[dim];
            continue;
        }

        if (shape1[dim] == 1)
        {
            // Broadcast shape1 to shape0
            int neg_dim = static_cast<int>(dim) - static_cast<int>(shape1.size());
            broadcast.push_back(graphlib::DimBroadcast(1, neg_dim, shape0[dim]));
            output_shape[dim] = shape0[dim];
        }
        else
        {
            TT_ASSERT(
                shape0[dim] == 1,
                "Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to "
                "broadcast");
            // Broadcast shape0 to shape1
            int neg_dim = static_cast<int>(dim) - static_cast<int>(shape0.size());
            broadcast.push_back(graphlib::DimBroadcast(0, neg_dim, shape1[dim]));
            output_shape[dim] = shape1[dim];
        }
    }

    return std::make_tuple(graphlib::Shape::create(output_shape), broadcast);
}

tt::graphlib::NodeContext reduce_broadcast_dimensions(
    tt::autograd::autograd_context &ac,
    const tt::graphlib::NodeContext &gradient,
    const tt::graphlib::Shape &input_shape,
    const tt::graphlib::Shape &grad_shape)
{
    // If shapes match, no reduction needed
    if (input_shape == grad_shape)
    {
        return gradient;
    }

    // Shapes don't match, we need to reduce along broadcast dimensions
    tt::graphlib::NodeContext result_grad = gradient;
    auto input_dims = input_shape.as_vector();
    auto grad_dims = grad_shape.as_vector();

    // Pad shapes with 1s at the beginning to match max rank
    size_t max_dims = std::max(input_dims.size(), grad_dims.size());

    std::vector<int> padded_input_dims(max_dims, 1);
    for (size_t i = 0; i < input_dims.size(); i++)
    {
        padded_input_dims[max_dims - input_dims.size() + i] = input_dims[i];
    }

    std::vector<int> padded_grad_dims(max_dims, 1);
    for (size_t i = 0; i < grad_dims.size(); i++)
    {
        padded_grad_dims[max_dims - grad_dims.size() + i] = grad_dims[i];
    }

    // For each dimension, if input_dim < grad_dim, we need to reduce_sum
    for (size_t i = 0; i < max_dims; i++)
    {
        if (padded_input_dims[i] >= padded_grad_dims[i])
            continue;

        int dim = static_cast<int>(i);
        result_grad = ac.autograd->create_op(
            ac, graphlib::OpType("reduce_sum", {}, {{"keep_dim", true}, {"dim_arg", dim}}), {result_grad});
    }

    return result_grad;
}

long initial_flops_estimate_output_dim(std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape_tuple)
{
    graphlib::Shape out_shape = std::get<0>(shape_tuple);
    return std::accumulate(out_shape.begin(), out_shape.end(), 1L, std::multiplies<int64_t>());
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> reduce_ops_shape(
    const Op &op, const std::vector<std::vector<std::uint32_t>> &in_shapes)
{
    int dim = op.attr_as<std::vector<int>>("dim_arg")[0];
    bool keep_dim = op.attr_as<bool>("keep_dim");

    if (dim < 0)
        dim += in_shapes[0].size();

    TT_ASSERT(dim >= 0 && dim < static_cast<int>(in_shapes[0].size()), "Reduce ops should have valid dim.");

    std::vector<std::uint32_t> ret = in_shapes[0];

    if (keep_dim)
        ret[dim] = 1;
    else
        ret.erase(ret.begin() + dim);

    return {graphlib::Shape::create(ret), {}};
}

std::vector<at::Tensor> promote_floating_dtypes(const std::vector<at::Tensor> &tensors)
{
    std::vector<at::Tensor> result;
    result.reserve(tensors.size());

    at::ScalarType promote_t = torch::kU8;
    for (const auto &t : tensors)
        if (t.is_floating_point())
            promote_t = at::promote_types(promote_t, t.scalar_type());

    for (const auto &t : tensors)
        if (t.is_floating_point() && t.scalar_type() != promote_t)
            result.emplace_back(t.to(promote_t));
        else
            result.emplace_back(t.clone());

    return result;
}

void decompose_nearest_interpolation(tt::DecomposingContext &dc, const tt::graphlib::NodeContext &activation, std::vector<int> sizes, bool channel_last)
{
    NodeContext result = activation;
    graphlib::Shape act_shape = activation.shape;
    const size_t rank = act_shape.size();
    if (rank == 3)
    {
        size_t c_dim = channel_last ? rank - 1 : rank - 2;
        size_t w_dim = channel_last ? rank - 2 : rank - 1;
        int input_c = static_cast<int>(act_shape[c_dim]);
        int input_w = static_cast<int>(act_shape[w_dim]);
        int size_w = sizes[0];
        int g_w = std::gcd(size_w, input_w);
        int up_w = size_w / g_w;
        int down_w = input_w / g_w;
        if (up_w > 1)
            result = dc.op(tt::graphlib::OpType("repeat_interleave", {}, {{"repeats", up_w}, {"dim", static_cast<int>(w_dim)}}), {result});
        if (down_w != 1)
        {
            result = dc.op(tt::graphlib::OpType("unsqueeze", {}, {{"dim", static_cast<int>(w_dim)}}), {result});
            std::vector<int64_t> weight_shape = {input_c, 1, 1, down_w};
            at::ScalarType datatype = tt::graphlib::data_format_to_scalar_type(result.output_df);
            at::Tensor weight_tensor = torch::zeros(weight_shape, torch::TensorOptions().dtype(datatype));
            weight_tensor.index_put_({torch::indexing::Slice(), 0, 0, 0}, 1.0);
            NodeContext weight = dc.tensor(weight_tensor);
            std::vector<int> stride = {1, down_w};
            std::vector<int> padding = {0, 0, 0, 0};
            std::vector<int> dilation = {1, 1};
            result = dc.op(tt::graphlib::OpType("conv2d", {}, {{"stride", stride}, {"groups", input_c}, {"padding", padding}, {"dilation", dilation}, {"channel_last", channel_last}}), {result, weight});
            result = dc.op(tt::graphlib::OpType("squeeze", {}, {{"dim", static_cast<int>(w_dim)}}), {result});
        }
        dc.fuse(result);
    }
    else if (rank == 4)
    {
        size_t c_dim = channel_last ? rank - 1 : rank - 3;
        size_t h_dim = channel_last ? rank - 3 : rank - 2;
        size_t w_dim = channel_last ? rank - 2 : rank - 1;
        int input_c = static_cast<int>(act_shape[c_dim]);
        int input_h = static_cast<int>(act_shape[h_dim]);
        int input_w = static_cast<int>(act_shape[w_dim]);
        int size_h = sizes[0];
        int size_w = sizes[1];
        int g_h = std::gcd(size_h, input_h);
        int g_w = std::gcd(size_w, input_w);
        int up_h = size_h / g_h;
        int up_w = size_w / g_w;
        int down_h = input_h / g_h;
        int down_w = input_w / g_w;
        NodeContext result = activation;
        if (up_h > 1)
            result = dc.op(tt::graphlib::OpType("repeat_interleave", {}, {{"repeats", up_h}, {"dim", static_cast<int>(h_dim)}}), {result});
        if (up_w > 1)
            result = dc.op(tt::graphlib::OpType("repeat_interleave", {}, {{"repeats", up_w}, {"dim", static_cast<int>(w_dim)}}), {result});
        if (down_w != 1)
        {
            std::vector<int64_t> weight_shape = {input_c, 1, 1, down_w};
            at::ScalarType datatype = tt::graphlib::data_format_to_scalar_type(result.output_df);
            at::Tensor weight_tensor = torch::zeros(weight_shape, torch::TensorOptions().dtype(datatype));
            weight_tensor.index_put_({torch::indexing::Slice(), 0, 0, 0}, 1.0);
            NodeContext weight = dc.tensor(weight_tensor);
            std::vector<int> stride = {1, down_w};
            std::vector<int> padding = {0, 0, 0, 0};
            std::vector<int> dilation = {1, 1};
            result = dc.op(tt::graphlib::OpType("conv2d", {}, {{"stride", stride}, {"groups", input_c}, {"padding", padding}, {"dilation", dilation}, {"channel_last", channel_last}}), {result, weight});
        }
        if (down_h != 1)
        {
            std::vector<int64_t> weight_shape = {input_c, 1, down_h, 1};
            at::ScalarType datatype = tt::graphlib::data_format_to_scalar_type(result.output_df);
            at::Tensor weight_tensor = torch::zeros(weight_shape, torch::TensorOptions().dtype(datatype));
            weight_tensor.index_put_({torch::indexing::Slice(), 0, 0, 0}, 1.0);
            NodeContext weight = dc.tensor(weight_tensor);
            std::vector<int> stride = {down_h, 1};
            std::vector<int> padding = {0, 0, 0, 0};
            std::vector<int> dilation = {1, 1};
            result = dc.op(tt::graphlib::OpType("conv2d", {}, {{"stride", stride}, {"groups", input_c}, {"padding", padding}, {"dilation", dilation},  {"channel_last",channel_last}}), {result, weight});
        }
        dc.fuse(result);
    }
    else
    {
        TT_THROW("Nearest interpolation is supported only for 3d and 4d inputs");
        unreachable();
    }
}

}  // namespace op_common
}  // namespace ops
}  // namespace tt
