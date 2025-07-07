// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/pybind11.h>
#pragma clang diagnostic pop

#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utils/logger.hpp>

#include "autograd/autograd.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"

namespace tt
{
namespace ops
{

namespace py = pybind11;

/**
 * In transition period we need mapping from new to the old op type, in order to preserve old functionalities.
 */
class NewToOldOpType
{
   public:
    NewToOldOpType()
    {
        mapping_[OpType::Abs] = "abs";
        mapping_[OpType::AdaptiveMaxPool2d] = "adaptive_max_pool2d";
        mapping_[OpType::Add] = "add";
        mapping_[OpType::AdvIndex] = "adv_index";
        mapping_[OpType::Argmax] = "argmax";
        mapping_[OpType::Atan] = "atan";
        mapping_[OpType::AvgPool1d] = "avg_pool1d";
        mapping_[OpType::AvgPool2d] = "avg_pool2d";
        mapping_[OpType::AvgPool3d] = "avg_pool3d";
        mapping_[OpType::Batchnorm] = "batchnorm";
        mapping_[OpType::Broadcast] = "broadcast";
        mapping_[OpType::Buffer] = "buffer";
        mapping_[OpType::Cast] = "cast";
        mapping_[OpType::Clip] = "clip";
        mapping_[OpType::Concatenate] = "concatenate";
        mapping_[OpType::Constant] = "constant";
        mapping_[OpType::Conv2d] = "conv2d";
        mapping_[OpType::Conv2dDepthwiseWeights] = "conv2d_depthwise_weights";
        mapping_[OpType::Conv2dDepthwiseWeightsBw] = "conv2d_depthwise_weights_bw";
        mapping_[OpType::Conv2dGroupedWeights] = "conv2d_grouped_weights";
        mapping_[OpType::Conv2dGroupedWeightsBw] = "conv2d_grouped_weights_bw";
        mapping_[OpType::Conv2dPrestrideAct] = "conv2d_prestride_act";
        mapping_[OpType::Conv2dPrestrideWeights] = "conv2d_prestride_weights";
        mapping_[OpType::Conv2dTranspose] = "conv2d_transpose";
        mapping_[OpType::Conv3d] = "conv3d";
        mapping_[OpType::ConvSum] = "conv_sum";
        mapping_[OpType::Cosine] = "cosine";
        mapping_[OpType::CumulativeSum] = "cumsum";
        mapping_[OpType::Depthwise] = "depthwise";
        mapping_[OpType::Dequantize] = "dequantize";
        mapping_[OpType::Divide] = "divide";
        mapping_[OpType::Downsample2d] = "downsample2d";
        mapping_[OpType::DramQueue] = "dram_queue";
        mapping_[OpType::Dropout] = "dropout";
        mapping_[OpType::Embedding] = "embedding";
        mapping_[OpType::EmbeddingBw] = "embedding_bw";
        mapping_[OpType::Equal] = "equal";
        mapping_[OpType::Erf] = "erf";
        mapping_[OpType::EthernetDatacopy] = "ethernet_datacopy";
        mapping_[OpType::Exp] = "exp";
        mapping_[OpType::FillCache] = "fill_cache";
        mapping_[OpType::ForgeDequantize] = "forge_dequantize";
        mapping_[OpType::ForgePad] = "forge_pad";
        mapping_[OpType::ForgeQuantize] = "forge_quantize";
        mapping_[OpType::ForgeRequantize] = "forge_requantize";
        mapping_[OpType::ForgeUnpad] = "forge_unpad";
        mapping_[OpType::Gather] = "gather";
        mapping_[OpType::Gelu] = "gelu";
        mapping_[OpType::GeluDerivative] = "gelu_derivative";
        mapping_[OpType::Greater] = "greater";
        mapping_[OpType::GreaterEqual] = "greater_equal";
        mapping_[OpType::GroupedReduceAvg] = "grouped_reduce_avg";
        mapping_[OpType::Heaviside] = "heaviside";
        mapping_[OpType::Hslice] = "hslice";
        mapping_[OpType::Hstack] = "hstack";
        mapping_[OpType::Index] = "index";
        mapping_[OpType::IndexCopy] = "index_copy";
        mapping_[OpType::Interleave] = "interleave";
        mapping_[OpType::Layernorm] = "layernorm";
        mapping_[OpType::LayernormBw] = "layernorm_bw";
        mapping_[OpType::LeakyRelu] = "leaky_relu";
        mapping_[OpType::Less] = "less";
        mapping_[OpType::LessEqual] = "less_equal";
        mapping_[OpType::Log] = "log";
        mapping_[OpType::LogSoftmax] = "log_softmax";
        mapping_[OpType::LogicalAnd] = "logical_and";
        mapping_[OpType::LogicalNot] = "logical_not";
        mapping_[OpType::Mask] = "mask";
        mapping_[OpType::Matmul] = "matmul";
        mapping_[OpType::MaxPool1d] = "max_pool1d";
        mapping_[OpType::MaxPool2d] = "max_pool2d";
        mapping_[OpType::MaxPool3d] = "max_pool3d";
        mapping_[OpType::Maximum] = "maximum";
        mapping_[OpType::Minimum] = "minimum";
        mapping_[OpType::Multiply] = "multiply";
        mapping_[OpType::Narrow] = "narrow";
        mapping_[OpType::Nop] = "nop";
        mapping_[OpType::NotEqual] = "not_equal";
        mapping_[OpType::Pad] = "pad";
        mapping_[OpType::PadTile] = "pad_tile";
        mapping_[OpType::PixelShuffle] = "pixel_shuffle";
        mapping_[OpType::Pow] = "pow";
        mapping_[OpType::Power] = "power";
        mapping_[OpType::Quantize] = "quantize";
        mapping_[OpType::Reciprocal] = "reciprocal";
        mapping_[OpType::ReduceAvg] = "reduce_avg";
        mapping_[OpType::ReduceMax] = "reduce_max";
        mapping_[OpType::ReduceSum] = "reduce_sum";
        mapping_[OpType::Relu] = "relu";
        mapping_[OpType::Remainder] = "remainder";
        mapping_[OpType::Repeat] = "repeat";
        mapping_[OpType::RepeatInterleave] = "repeat_interleave";
        mapping_[OpType::Requantize] = "requantize";
        mapping_[OpType::Reshape] = "reshape";
        mapping_[OpType::Resize1d] = "resize1d";
        mapping_[OpType::Resize2d] = "resize2d";
        mapping_[OpType::Resize3d] = "resize3d";
        mapping_[OpType::Select] = "select";
        mapping_[OpType::Sigmoid] = "sigmoid";
        mapping_[OpType::Sine] = "sine";
        mapping_[OpType::Softmax] = "softmax";
        mapping_[OpType::SoftmaxBw] = "softmax_bw";
        mapping_[OpType::SparseMatmul] = "sparse_matmul";
        mapping_[OpType::Sqrt] = "sqrt";
        mapping_[OpType::Squeeze] = "squeeze";
        mapping_[OpType::Stack] = "stack";
        mapping_[OpType::Subtract] = "subtract";
        mapping_[OpType::Tanh] = "tanh";
        mapping_[OpType::TileBroadcast] = "tile_broadcast";
        mapping_[OpType::Tilizer] = "tilizer";
        mapping_[OpType::Transpose] = "transpose";
        mapping_[OpType::Unsqueeze] = "unsqueeze";
        mapping_[OpType::UpdateCache] = "update_cache";
        mapping_[OpType::Upsample2d] = "upsample2d";
        mapping_[OpType::Vslice] = "vslice";
        mapping_[OpType::Vstack] = "vstack";
        mapping_[OpType::Where] = "where";
    }

    const std::string &operator[](OpType op_type) const { return mapping_.at(op_type); }

   private:
    std::unordered_map<OpType, std::string> mapping_;
};

/**
 * In transition period we need mapping from old to the new op type, in order to preserve old functionalities.
 */
class OldToNewOpType
{
   public:
    OldToNewOpType()
    {
        mapping_["abs"] = OpType::Abs;
        mapping_["adaptive_max_pool2d"] = OpType::AdaptiveMaxPool2d;
        mapping_["add"] = OpType::Add;
        mapping_["adv_index"] = OpType::AdvIndex;
        mapping_["argmax"] = OpType::Argmax;
        mapping_["atan"] = OpType::Atan;
        mapping_["avg_pool1d"] = OpType::AvgPool1d;
        mapping_["avg_pool2d"] = OpType::AvgPool2d;
        mapping_["avg_pool3d"] = OpType::AvgPool3d;
        mapping_["batchnorm"] = OpType::Batchnorm;
        mapping_["broadcast"] = OpType::Broadcast;
        mapping_["buffer"] = OpType::Buffer;
        mapping_["cast"] = OpType::Cast;
        mapping_["clip"] = OpType::Clip;
        mapping_["concatenate"] = OpType::Concatenate;
        mapping_["constant"] = OpType::Constant;
        mapping_["conv2d"] = OpType::Conv2d;
        mapping_["conv2d_depthwise_weights"] = OpType::Conv2dDepthwiseWeights;
        mapping_["conv2d_depthwise_weights_bw"] = OpType::Conv2dDepthwiseWeightsBw;
        mapping_["conv2d_grouped_weights"] = OpType::Conv2dGroupedWeights;
        mapping_["conv2d_grouped_weights_bw"] = OpType::Conv2dGroupedWeightsBw;
        mapping_["conv2d_prestride_act"] = OpType::Conv2dPrestrideAct;
        mapping_["conv2d_prestride_weights"] = OpType::Conv2dPrestrideWeights;
        mapping_["conv2d_transpose"] = OpType::Conv2dTranspose;
        mapping_["conv3d"] = OpType::Conv3d;
        mapping_["conv_sum"] = OpType::ConvSum;
        mapping_["cosine"] = OpType::Cosine;
        mapping_["cumsum"] = OpType::CumulativeSum;
        mapping_["depthwise"] = OpType::Depthwise;
        mapping_["dequantize"] = OpType::Dequantize;
        mapping_["divide"] = OpType::Divide;
        mapping_["downsample2d"] = OpType::Downsample2d;
        mapping_["dram_queue"] = OpType::DramQueue;
        mapping_["dropout"] = OpType::Dropout;
        mapping_["embedding"] = OpType::Embedding;
        mapping_["embedding_bw"] = OpType::EmbeddingBw;
        mapping_["equal"] = OpType::Equal;
        mapping_["erf"] = OpType::Erf;
        mapping_["ethernet_datacopy"] = OpType::EthernetDatacopy;
        mapping_["exp"] = OpType::Exp;
        mapping_["fill_cache"] = OpType::FillCache;
        mapping_["forge_dequantize"] = OpType::ForgeDequantize;
        mapping_["forge_pad"] = OpType::ForgePad;
        mapping_["forge_quantize"] = OpType::ForgeQuantize;
        mapping_["forge_requantize"] = OpType::ForgeRequantize;
        mapping_["forge_unpad"] = OpType::ForgeUnpad;
        mapping_["gather"] = OpType::Gather;
        mapping_["gelu"] = OpType::Gelu;
        mapping_["gelu_derivative"] = OpType::GeluDerivative;
        mapping_["greater"] = OpType::Greater;
        mapping_["greater_equal"] = OpType::GreaterEqual;
        mapping_["grouped_reduce_avg"] = OpType::GroupedReduceAvg;
        mapping_["heaviside"] = OpType::Heaviside;
        mapping_["hslice"] = OpType::Hslice;
        mapping_["hstack"] = OpType::Hstack;
        mapping_["index"] = OpType::Index;
        mapping_["index_copy"] = OpType::IndexCopy;
        mapping_["interleave"] = OpType::Interleave;
        mapping_["layernorm"] = OpType::Layernorm;
        mapping_["layernorm_bw"] = OpType::LayernormBw;
        mapping_["leaky_relu"] = OpType::LeakyRelu;
        mapping_["less"] = OpType::Less;
        mapping_["less_equal"] = OpType::LessEqual;
        mapping_["log"] = OpType::Log;
        mapping_["log_softmax"] = OpType::LogSoftmax;
        mapping_["logical_and"] = OpType::LogicalAnd;
        mapping_["logical_not"] = OpType::LogicalNot;
        mapping_["mask"] = OpType::Mask;
        mapping_["matmul"] = OpType::Matmul;
        mapping_["max_pool1d"] = OpType::MaxPool1d;
        mapping_["max_pool2d"] = OpType::MaxPool2d;
        mapping_["max_pool3d"] = OpType::MaxPool3d;
        mapping_["maximum"] = OpType::Maximum;
        mapping_["minimum"] = OpType::Minimum;
        mapping_["multiply"] = OpType::Multiply;
        mapping_["narrow"] = OpType::Narrow;
        mapping_["nop"] = OpType::Nop;
        mapping_["not_equal"] = OpType::NotEqual;
        mapping_["pad"] = OpType::Pad;
        mapping_["pad_tile"] = OpType::PadTile;
        mapping_["pixel_shuffle"] = OpType::PixelShuffle;
        mapping_["pow"] = OpType::Pow;
        mapping_["power"] = OpType::Power;
        mapping_["quantize"] = OpType::Quantize;
        mapping_["reciprocal"] = OpType::Reciprocal;
        mapping_["reduce_avg"] = OpType::ReduceAvg;
        mapping_["reduce_max"] = OpType::ReduceMax;
        mapping_["reduce_sum"] = OpType::ReduceSum;
        mapping_["relu"] = OpType::Relu;
        mapping_["remainder"] = OpType::Remainder;
        mapping_["repeat"] = OpType::Repeat;
        mapping_["repeat_interleave"] = OpType::RepeatInterleave;
        mapping_["requantize"] = OpType::Requantize;
        mapping_["reshape"] = OpType::Reshape;
        mapping_["resize1d"] = OpType::Resize1d;
        mapping_["resize2d"] = OpType::Resize2d;
        mapping_["resize3d"] = OpType::Resize3d;
        mapping_["select"] = OpType::Select;
        mapping_["sigmoid"] = OpType::Sigmoid;
        mapping_["sine"] = OpType::Sine;
        mapping_["softmax"] = OpType::Softmax;
        mapping_["softmax_bw"] = OpType::SoftmaxBw;
        mapping_["sparse_matmul"] = OpType::SparseMatmul;
        mapping_["sqrt"] = OpType::Sqrt;
        mapping_["squeeze"] = OpType::Squeeze;
        mapping_["stack"] = OpType::Stack;
        mapping_["subtract"] = OpType::Subtract;
        mapping_["tanh"] = OpType::Tanh;
        mapping_["tile_broadcast"] = OpType::TileBroadcast;
        mapping_["tilizer"] = OpType::Tilizer;
        mapping_["transpose"] = OpType::Transpose;
        mapping_["unsqueeze"] = OpType::Unsqueeze;
        mapping_["update_cache"] = OpType::UpdateCache;
        mapping_["upsample2d"] = OpType::Upsample2d;
        mapping_["vslice"] = OpType::Vslice;
        mapping_["vstack"] = OpType::Vstack;
        mapping_["where"] = OpType::Where;
    }

    OpType operator[](const std::string &old_op_type) const { return mapping_.at(old_op_type); }

   private:
    std::unordered_map<std::string, OpType> mapping_;
};

static NewToOldOpType new_to_old_op_type_mapper;
static OldToNewOpType old_to_new_op_type_mapper;

Op::Op(const graphlib::OpType &old_op_type) :
    type_(old_to_new_op_type_mapper[old_op_type.op]), attrs_(old_op_type.named_attrs)
{
}

const std::string &Op::as_string() const { return new_to_old_op_type_mapper[type_]; }

/* ------------------------------------------------------------------------------------------------------------------*
 * Default implementation for ops that are not cpp implemented yet. We will invoke old python code to evaluate them. *
 * ------------------------------------------------------------------------------------------------------------------*/

at::Tensor Op::base_eval(const graphlib::OpType &old_op_type, const std::vector<at::Tensor> &tensors) const
{
    py::function eval = py::module_::import("forge.op.eval.forge").attr("get_f_forge_eval")(&old_op_type);
    return eval(&tensors).cast<at::Tensor>();
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::base_shape(
    const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    py::function shape = py::module_::import("forge.op.eval.forge").attr("get_f_forge_shape")(&old_op_type);
    py::tuple result = shape(&inputs);
    if (result.size() != 2)
        throw std::runtime_error("Expected a tuple of shape and broadcast.");

    return std::make_tuple(
        graphlib::Shape::create(result[0].cast<std::vector<std::uint32_t>>()),
        result[1].cast<std::vector<graphlib::DimBroadcast>>());
}

tt::graphlib::NodeContext Op::base_backward(
    const graphlib::OpType &old_op_type,
    tt::autograd::autograd_context &context,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient) const
{
    py::function backward = py::module_::import("forge.op.eval.forge").attr("get_f_forge_backward")(&old_op_type);
    return backward(&context, operand, &inputs, &output, &gradient).cast<tt::graphlib::NodeContext>();
}

void Op::base_decompose(
    const graphlib::OpType &old_op_type,
    const char *dispatch,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const
{
    py::function decompose = py::module_::import("forge.op.eval.forge").attr(dispatch)(&old_op_type);
    decompose(&dc, &inputs);
}

long Op::base_initial_flops_estimate(
    const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    py::function initial_flops_estimate =
        py::module_::import("forge.op.eval.forge").attr("get_f_forge_initial_flops_estimate")(&old_op_type);
    py::object ret = initial_flops_estimate(&inputs);

    return ret.is_none() ? 0 : ret.cast<long>();
}

bool Op::base_is_tm(const graphlib::OpType &old_op_type) const
{
    py::function is_tm = py::module_::import("forge.op.eval.forge").attr("is_tm");
    return is_tm(&old_op_type).cast<bool>();
}

bool Op::base_is_eltwise(const graphlib::OpType &old_op_type) const
{
    py::function is_eltwise = py::module_::import("forge.op.eval.forge").attr("is_eltwise");
    return is_eltwise(&old_op_type).cast<bool>();
}

bool Op::base_is_eltwise_unary(const graphlib::OpType &old_op_type) const
{
    py::function is_eltwise_unary = py::module_::import("forge.op.eval.forge").attr("is_eltwise_unary");
    return is_eltwise_unary(&old_op_type).cast<bool>();
}

bool Op::base_is_eltwise_binary(const graphlib::OpType &old_op_type) const
{
    py::function is_eltwise_binary = py::module_::import("forge.op.eval.forge").attr("is_eltwise_binary");
    return is_eltwise_binary(&old_op_type).cast<bool>();
}

bool Op::base_is_eltwise_nary(const graphlib::OpType &old_op_type) const
{
    py::function is_eltwise_nary = py::module_::import("forge.op.eval.forge").attr("is_eltwise_nary");
    return is_eltwise_nary(&old_op_type).cast<bool>();
}

/* ------------------------------*
 * Dispatching based on op type. *
 * ------------------------------*/

at::Tensor Op::eval(const graphlib::OpType &old_op_type, const std::vector<at::Tensor> &tensors) const
{
    switch (type_)
    {
        case OpType::Abs: return abs::eval(*this, tensors);
        case OpType::Constant: return constant::eval(*this, tensors);
        case OpType::Multiply: return multiply::eval(*this, tensors);
        default: return base_eval(old_op_type, tensors);
    }
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::shape(
    const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    switch (type_)
    {
        case OpType::Abs: return abs::shape(*this, inputs);
        case OpType::Constant: return constant::shape(*this, inputs);
        case OpType::Multiply: return multiply::shape(*this, inputs);
        default: return base_shape(old_op_type, inputs);
    }
}

tt::graphlib::NodeContext Op::backward(
    const graphlib::OpType &old_op_type,
    tt::autograd::autograd_context &context,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient) const
{
    switch (type_)
    {
        case OpType::Abs: return abs::backward(*this, context, operand, inputs, output, gradient);
        case OpType::Constant: return constant::backward(*this, context, operand, inputs, output, gradient);
        case OpType::Multiply: return multiply::backward(*this, context, operand, inputs, output, gradient);
        default: return base_backward(old_op_type, context, operand, inputs, output, gradient);
    }
}

/**
 * Note: We will most likely get rid of distinct implementations for decompose, once we investigate why they even exist.
 * They are needed for now in order to unblock ops migration from python to cpp.
 */
template <DecomposeEpoch epoch>
void Op::decompose(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const
{
    if constexpr (epoch == DecomposeEpoch::Initial)
        return decompose_initial(old_op_type, dc, inputs);
    else if constexpr (epoch == DecomposeEpoch::PostOptimize)
        return decompose_post_optimize(old_op_type, dc, inputs);
    else if constexpr (epoch == DecomposeEpoch::PostAutograd)
        return decompose_post_autograd(old_op_type, dc, inputs);
    else
        static_assert("Invalid decomposing epoch.");
}

void Op::decompose_initial(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const
{
    switch (type_)
    {
        case OpType::Abs: return;
        case OpType::Constant: return;
        case OpType::Multiply: return;
        default: return base_decompose(old_op_type, "get_f_forge_decompose", dc, inputs);
    }
}

void Op::decompose_post_optimize(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const
{
    switch (type_)
    {
        case OpType::Abs: return;
        case OpType::Constant: return;
        case OpType::Multiply: return;
        default: return base_decompose(old_op_type, "get_f_forge_decompose_post_optimize", dc, inputs);
    }
}

void Op::decompose_post_autograd(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const
{
    switch (type_)
    {
        case OpType::Abs: return;
        case OpType::Constant: return;
        case OpType::Multiply: return multiply::decompose_post_autograd(*this, dc, inputs);
        default: return base_decompose(old_op_type, "get_f_forge_decompose_post_autograd", dc, inputs);
    }
}

long Op::initial_flops_estimate(
    const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    switch (type_)
    {
        case OpType::Abs: return abs::initial_flops_estimate(*this, inputs);
        case OpType::Constant: return 0;
        case OpType::Multiply: return 0;
        default: return base_initial_flops_estimate(old_op_type, inputs);
    }
}

bool Op::is_tm(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return false;
        case OpType::Constant: return false;
        case OpType::Multiply: return false;
        default: return base_is_tm(old_op_type);
    }
}

bool Op::is_eltwise(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return true;
        case OpType::Constant: return false;
        case OpType::Multiply: return true;
        default: return base_is_eltwise(old_op_type);
    }
}

bool Op::is_eltwise_unary(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return true;
        case OpType::Constant: return false;
        case OpType::Multiply: return false;
        default: return base_is_eltwise_unary(old_op_type);
    }
}

bool Op::is_eltwise_binary(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return false;
        case OpType::Constant: return false;
        case OpType::Multiply: return true;
        default: return base_is_eltwise_binary(old_op_type);
    }
}
bool Op::is_eltwise_nary(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return false;
        case OpType::Constant: return false;
        case OpType::Multiply: return true;
        default: return base_is_eltwise_nary(old_op_type);
    }
}

/**
 * Explicit instantiations to enable pybind symbol resolution.
 */
template void Op::decompose<DecomposeEpoch::Initial>(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const;

template void Op::decompose<DecomposeEpoch::PostOptimize>(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const;

template void Op::decompose<DecomposeEpoch::PostAutograd>(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const;

}  // namespace ops
}  // namespace tt
