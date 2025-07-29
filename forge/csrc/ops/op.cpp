// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op.hpp"

#include <utils/assert.hpp>

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
        mapping_[OpType::Greater] = "greater";
        mapping_[OpType::GreaterEqual] = "greater_equal";
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
        mapping_["greater"] = OpType::Greater;
        mapping_["greater_equal"] = OpType::GreaterEqual;
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
    type_(old_to_new_op_type_mapper[old_op_type.op_]), attrs_(old_op_type.named_attrs_)
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

/* ------------------------------*
 * Dispatching based on op type. *
 * ------------------------------*/

at::Tensor Op::eval(const graphlib::OpType &old_op_type, const std::vector<at::Tensor> &tensors) const
{
    switch (type_)  // clang-format off
    {
        case OpType::Abs: return abs::eval(old_op_type, *this, tensors);
        case OpType::AdaptiveMaxPool2d: return adaptive_max_pool_2d::eval(old_op_type, *this, tensors);
        case OpType::Add: return add::eval(old_op_type, *this, tensors);
        case OpType::AdvIndex: return adv_index::eval(old_op_type, *this, tensors);
        case OpType::Argmax: return argmax::eval(old_op_type, *this, tensors);
        case OpType::Atan: return atan::eval(old_op_type, *this, tensors);
        case OpType::AvgPool1d: return avg_pool_1d::eval(old_op_type, *this, tensors);
        case OpType::AvgPool2d: return avg_pool_2d::eval(old_op_type, *this, tensors);
        case OpType::AvgPool3d: return avg_pool_3d::eval(old_op_type, *this, tensors);
        case OpType::Batchnorm: return batchnorm::eval(old_op_type, *this, tensors);
        case OpType::Broadcast: return broadcast::eval(old_op_type, *this, tensors);
        case OpType::Buffer: return buffer::eval(old_op_type, *this, tensors);
        case OpType::Cast: return cast::eval(old_op_type, *this, tensors);
        case OpType::Clip: return clip::eval(old_op_type, *this, tensors);
        case OpType::Concatenate: return concatenate::eval(old_op_type, *this, tensors);
        case OpType::Constant: return constant::eval(old_op_type, *this, tensors);
        case OpType::Conv2d: return conv_2d::eval(old_op_type, *this, tensors);
        case OpType::Conv2dDepthwiseWeights: return conv_2d_depthwise_weights::eval(old_op_type, *this, tensors);
        case OpType::Conv2dDepthwiseWeightsBw: return conv_2d_depthwise_weights_bw::eval(old_op_type, *this, tensors);
        case OpType::Conv2dGroupedWeights: return conv_2d_grouped_weights::eval(old_op_type, *this, tensors);
        case OpType::Conv2dGroupedWeightsBw: return conv_2d_grouped_weights_bw::eval(old_op_type, *this, tensors);
        case OpType::Conv2dPrestrideAct: return conv_2d_prestride_act::eval(old_op_type, *this, tensors);
        case OpType::Conv2dPrestrideWeights: return conv_2d_prestride_weights::eval(old_op_type, *this, tensors);
        case OpType::Conv2dTranspose: return conv_2d_transpose::eval(old_op_type, *this, tensors);
        case OpType::Conv3d: return conv_3d::eval(old_op_type, *this, tensors);
        case OpType::ConvSum: return conv_sum::eval(old_op_type, *this, tensors);
        case OpType::Cosine: return cosine::eval(old_op_type, *this, tensors);
        case OpType::CumulativeSum: return cumulative_sum::eval(old_op_type, *this, tensors);
        case OpType::Depthwise: return depthwise::eval(old_op_type, *this, tensors);
        case OpType::Dequantize: return dequantize::eval(old_op_type, *this, tensors);
        case OpType::Divide: return divide::eval(old_op_type, *this, tensors);
        case OpType::Downsample2d: return downsample_2d::eval(old_op_type, *this, tensors);
        case OpType::DramQueue: return dram_queue::eval(old_op_type, *this, tensors);
        case OpType::Dropout: return dropout::eval(old_op_type, *this, tensors);
        case OpType::Embedding: return embedding::eval(old_op_type, *this, tensors);
        case OpType::EmbeddingBw: return embedding_bw::eval(old_op_type, *this, tensors);
        case OpType::Equal: return equal::eval(old_op_type, *this, tensors);
        case OpType::Erf: return erf::eval(old_op_type, *this, tensors);
        case OpType::EthernetDatacopy: return ethernet_data_copy::eval(old_op_type, *this, tensors);
        case OpType::Exp: return exp::eval(old_op_type, *this, tensors);
        case OpType::FillCache: return fill_cache::eval(old_op_type, *this, tensors);
        case OpType::ForgeDequantize: return forge_dequantize::eval(old_op_type, *this, tensors);
        case OpType::ForgePad: return forge_pad::eval(old_op_type, *this, tensors);
        case OpType::ForgeQuantize: return forge_quantize::eval(old_op_type, *this, tensors);
        case OpType::ForgeRequantize: return forge_requantize::eval(old_op_type, *this, tensors);
        case OpType::ForgeUnpad: return forge_unpad::eval(old_op_type, *this, tensors);
        case OpType::Gather: return gather::eval(old_op_type, *this, tensors);
        case OpType::Gelu: return gelu::eval(old_op_type, *this, tensors);
        case OpType::Greater: return greater::eval(old_op_type, *this, tensors);
        case OpType::GreaterEqual: return greater_equal::eval(old_op_type, *this, tensors);
        case OpType::Heaviside: return heaviside::eval(old_op_type, *this, tensors);
        case OpType::Hslice: return hslice::eval(old_op_type, *this, tensors);
        case OpType::Hstack: return hstack::eval(old_op_type, *this, tensors);
        case OpType::Index: return index::eval(old_op_type, *this, tensors);
        case OpType::IndexCopy: return index_copy::eval(old_op_type, *this, tensors);
        case OpType::Interleave: return interleave::eval(old_op_type, *this, tensors);
        case OpType::Layernorm: return layernorm::eval(old_op_type, *this, tensors);
        case OpType::LayernormBw: return layernorm_bw::eval(old_op_type, *this, tensors);
        case OpType::LeakyRelu: return leaky_relu::eval(old_op_type, *this, tensors);
        case OpType::Less: return less::eval(old_op_type, *this, tensors);
        case OpType::LessEqual: return less_equal::eval(old_op_type, *this, tensors);
        case OpType::Log: return log::eval(old_op_type, *this, tensors);
        case OpType::LogSoftmax: return log_softmax::eval(old_op_type, *this, tensors);
        case OpType::LogicalAnd: return logical_and::eval(old_op_type, *this, tensors);
        case OpType::LogicalNot: return logical_not::eval(old_op_type, *this, tensors);
        case OpType::Mask: return mask::eval(old_op_type, *this, tensors);
        case OpType::Matmul: return matmul::eval(old_op_type, *this, tensors);
        case OpType::MaxPool1d: return max_pool_1d::eval(old_op_type, *this, tensors);
        case OpType::MaxPool2d: return max_pool_2d::eval(old_op_type, *this, tensors);
        case OpType::MaxPool3d: return max_pool_3d::eval(old_op_type, *this, tensors);
        case OpType::Maximum: return maximum::eval(old_op_type, *this, tensors);
        case OpType::Minimum: return minimum::eval(old_op_type, *this, tensors);
        case OpType::Multiply: return multiply::eval(old_op_type, *this, tensors);
        case OpType::Narrow: return narrow::eval(old_op_type, *this, tensors);
        case OpType::Nop: return nop::eval(old_op_type, *this, tensors);
        case OpType::NotEqual: return not_equal::eval(old_op_type, *this, tensors);
        case OpType::Pad: return pad::eval(old_op_type, *this, tensors);
        case OpType::PadTile: return pad_tile::eval(old_op_type, *this, tensors);
        case OpType::PixelShuffle: return pixel_shuffle::eval(old_op_type, *this, tensors);
        case OpType::Pow: return pow::eval(old_op_type, *this, tensors);
        case OpType::Power: return power::eval(old_op_type, *this, tensors);
        case OpType::Quantize: return quantize::eval(old_op_type, *this, tensors);
        case OpType::Reciprocal: return reciprocal::eval(old_op_type, *this, tensors);
        case OpType::ReduceAvg: return reduce_avg::eval(old_op_type, *this, tensors);
        case OpType::ReduceMax: return reduce_max::eval(old_op_type, *this, tensors);
        case OpType::ReduceSum: return reduce_sum::eval(old_op_type, *this, tensors);
        case OpType::Relu: return relu::eval(old_op_type, *this, tensors);
        case OpType::Remainder: return remainder::eval(old_op_type, *this, tensors);
        case OpType::Repeat: return repeat::eval(old_op_type, *this, tensors);
        case OpType::RepeatInterleave: return repeat_interleave::eval(old_op_type, *this, tensors);
        case OpType::Requantize: return requantize::eval(old_op_type, *this, tensors);
        case OpType::Reshape: return reshape::eval(old_op_type, *this, tensors);
        case OpType::Resize1d: return resize_1d::eval(old_op_type, *this, tensors);
        case OpType::Resize2d: return resize_2d::eval(old_op_type, *this, tensors);
        case OpType::Resize3d: return resize_3d::eval(old_op_type, *this, tensors);
        case OpType::Select: return select::eval(old_op_type, *this, tensors);
        case OpType::Sigmoid: return sigmoid::eval(old_op_type, *this, tensors);
        case OpType::Sine: return sine::eval(old_op_type, *this, tensors);
        case OpType::Softmax: return softmax::eval(old_op_type, *this, tensors);
        case OpType::SoftmaxBw: return softmax_bw::eval(old_op_type, *this, tensors);
        case OpType::SparseMatmul: return sparse_matmul::eval(old_op_type, *this, tensors);
        case OpType::Sqrt: return sqrt::eval(old_op_type, *this, tensors);
        case OpType::Squeeze: return squeeze::eval(old_op_type, *this, tensors);
        case OpType::Stack: return stack::eval(old_op_type, *this, tensors);
        case OpType::Subtract: return subtract::eval(old_op_type, *this, tensors);
        case OpType::Tanh: return tanh::eval(old_op_type, *this, tensors);
        case OpType::Tilizer: return tilizer::eval(old_op_type, *this, tensors);
        case OpType::Transpose: return transpose::eval(old_op_type, *this, tensors);
        case OpType::Unsqueeze: return unsqueeze::eval(old_op_type, *this, tensors);
        case OpType::UpdateCache: return update_cache::eval(old_op_type, *this, tensors);
        case OpType::Upsample2d: return upsample_2d::eval(old_op_type, *this, tensors);
        case OpType::Vslice: return vslice::eval(old_op_type, *this, tensors);
        case OpType::Vstack: return vstack::eval(old_op_type, *this, tensors);
        case OpType::Where: return where::eval(old_op_type, *this, tensors);
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }  // clang-format on
}

std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> Op::shape(
    const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    switch (type_)  // clang-format off
    {
        case OpType::Abs: return abs::shape(old_op_type, *this, inputs);
        case OpType::AdaptiveMaxPool2d: return adaptive_max_pool_2d::shape(old_op_type, *this, inputs);
        case OpType::Add: return add::shape(old_op_type, *this, inputs);
        case OpType::AdvIndex: return adv_index::shape(old_op_type, *this, inputs);
        case OpType::Argmax: return argmax::shape(old_op_type, *this, inputs);
        case OpType::Atan: return atan::shape(old_op_type, *this, inputs);
        case OpType::AvgPool1d: return avg_pool_1d::shape(old_op_type, *this, inputs);
        case OpType::AvgPool2d: return avg_pool_2d::shape(old_op_type, *this, inputs);
        case OpType::AvgPool3d: return avg_pool_3d::shape(old_op_type, *this, inputs);
        case OpType::Batchnorm: return batchnorm::shape(old_op_type, *this, inputs);
        case OpType::Broadcast: return broadcast::shape(old_op_type, *this, inputs);
        case OpType::Buffer: return buffer::shape(old_op_type, *this, inputs);
        case OpType::Cast: return cast::shape(old_op_type, *this, inputs);
        case OpType::Clip: return clip::shape(old_op_type, *this, inputs);
        case OpType::Concatenate: return concatenate::shape(old_op_type, *this, inputs);
        case OpType::Constant: return constant::shape(old_op_type, *this, inputs);
        case OpType::Conv2d: return conv_2d::shape(old_op_type, *this, inputs);
        case OpType::Conv2dDepthwiseWeights: return conv_2d_depthwise_weights::shape(old_op_type, *this, inputs);
        case OpType::Conv2dDepthwiseWeightsBw: return conv_2d_depthwise_weights_bw::shape(old_op_type, *this, inputs);
        case OpType::Conv2dGroupedWeights: return conv_2d_grouped_weights::shape(old_op_type, *this, inputs);
        case OpType::Conv2dGroupedWeightsBw: return conv_2d_grouped_weights_bw::shape(old_op_type, *this, inputs);
        case OpType::Conv2dPrestrideAct: return conv_2d_prestride_act::shape(old_op_type, *this, inputs);
        case OpType::Conv2dPrestrideWeights: return conv_2d_prestride_weights::shape(old_op_type, *this, inputs);
        case OpType::Conv2dTranspose: return conv_2d_transpose::shape(old_op_type, *this, inputs);
        case OpType::Conv3d: return conv_3d::shape(old_op_type, *this, inputs);
        case OpType::ConvSum: return conv_sum::shape(old_op_type, *this, inputs);
        case OpType::Cosine: return cosine::shape(old_op_type, *this, inputs);
        case OpType::CumulativeSum: return cumulative_sum::shape(old_op_type, *this, inputs);
        case OpType::Depthwise: return depthwise::shape(old_op_type, *this, inputs);
        case OpType::Dequantize: return dequantize::shape(old_op_type, *this, inputs);
        case OpType::Divide: return divide::shape(old_op_type, *this, inputs);
        case OpType::Downsample2d: return downsample_2d::shape(old_op_type, *this, inputs);
        case OpType::DramQueue: return dram_queue::shape(old_op_type, *this, inputs);
        case OpType::Dropout: return dropout::shape(old_op_type, *this, inputs);
        case OpType::Embedding: return embedding::shape(old_op_type, *this, inputs);
        case OpType::EmbeddingBw: return embedding_bw::shape(old_op_type, *this, inputs);
        case OpType::Equal: return equal::shape(old_op_type, *this, inputs);
        case OpType::Erf: return erf::shape(old_op_type, *this, inputs);
        case OpType::EthernetDatacopy: return ethernet_data_copy::shape(old_op_type, *this, inputs);
        case OpType::Exp: return exp::shape(old_op_type, *this, inputs);
        case OpType::FillCache: return fill_cache::shape(old_op_type, *this, inputs);
        case OpType::ForgeDequantize: return forge_dequantize::shape(old_op_type, *this, inputs);
        case OpType::ForgePad: return forge_pad::shape(old_op_type, *this, inputs);
        case OpType::ForgeQuantize: return forge_quantize::shape(old_op_type, *this, inputs);
        case OpType::ForgeRequantize: return forge_requantize::shape(old_op_type, *this, inputs);
        case OpType::ForgeUnpad: return forge_unpad::shape(old_op_type, *this, inputs);
        case OpType::Gather: return gather::shape(old_op_type, *this, inputs);
        case OpType::Gelu: return gelu::shape(old_op_type, *this, inputs);
        case OpType::Greater: return greater::shape(old_op_type, *this, inputs);
        case OpType::GreaterEqual: return greater_equal::shape(old_op_type, *this, inputs);
        case OpType::Heaviside: return heaviside::shape(old_op_type, *this, inputs);
        case OpType::Hslice: return hslice::shape(old_op_type, *this, inputs);
        case OpType::Hstack: return hstack::shape(old_op_type, *this, inputs);
        case OpType::Index: return index::shape(old_op_type, *this, inputs);
        case OpType::IndexCopy: return index_copy::shape(old_op_type, *this, inputs);
        case OpType::Interleave: return interleave::shape(old_op_type, *this, inputs);
        case OpType::Layernorm: return layernorm::shape(old_op_type, *this, inputs);
        case OpType::LayernormBw: return layernorm_bw::shape(old_op_type, *this, inputs);
        case OpType::LeakyRelu: return leaky_relu::shape(old_op_type, *this, inputs);
        case OpType::Less: return less::shape(old_op_type, *this, inputs);
        case OpType::LessEqual: return less_equal::shape(old_op_type, *this, inputs);
        case OpType::Log: return log::shape(old_op_type, *this, inputs);
        case OpType::LogSoftmax: return log_softmax::shape(old_op_type, *this, inputs);
        case OpType::LogicalAnd: return logical_and::shape(old_op_type, *this, inputs);
        case OpType::LogicalNot: return logical_not::shape(old_op_type, *this, inputs);
        case OpType::Mask: return mask::shape(old_op_type, *this, inputs);
        case OpType::Matmul: return matmul::shape(old_op_type, *this, inputs);
        case OpType::MaxPool1d: return max_pool_1d::shape(old_op_type, *this, inputs);
        case OpType::MaxPool2d: return max_pool_2d::shape(old_op_type, *this, inputs);
        case OpType::MaxPool3d: return max_pool_3d::shape(old_op_type, *this, inputs);
        case OpType::Maximum: return maximum::shape(old_op_type, *this, inputs);
        case OpType::Minimum: return minimum::shape(old_op_type, *this, inputs);
        case OpType::Multiply: return multiply::shape(old_op_type, *this, inputs);
        case OpType::Narrow: return narrow::shape(old_op_type, *this, inputs);
        case OpType::Nop: return nop::shape(old_op_type, *this, inputs);
        case OpType::NotEqual: return not_equal::shape(old_op_type, *this, inputs);
        case OpType::Pad: return pad::shape(old_op_type, *this, inputs);
        case OpType::PadTile: return pad_tile::shape(old_op_type, *this, inputs);
        case OpType::PixelShuffle: return pixel_shuffle::shape(old_op_type, *this, inputs);
        case OpType::Pow: return pow::shape(old_op_type, *this, inputs);
        case OpType::Power: return power::shape(old_op_type, *this, inputs);
        case OpType::Quantize: return quantize::shape(old_op_type, *this, inputs);
        case OpType::Reciprocal: return reciprocal::shape(old_op_type, *this, inputs);
        case OpType::ReduceAvg: return reduce_avg::shape(old_op_type, *this, inputs);
        case OpType::ReduceMax: return reduce_max::shape(old_op_type, *this, inputs);
        case OpType::ReduceSum: return reduce_sum::shape(old_op_type, *this, inputs);
        case OpType::Relu: return relu::shape(old_op_type, *this, inputs);
        case OpType::Remainder: return remainder::shape(old_op_type, *this, inputs);
        case OpType::Repeat: return repeat::shape(old_op_type, *this, inputs);
        case OpType::RepeatInterleave: return repeat_interleave::shape(old_op_type, *this, inputs);
        case OpType::Requantize: return requantize::shape(old_op_type, *this, inputs);
        case OpType::Reshape: return reshape::shape(old_op_type, *this, inputs);
        case OpType::Resize1d: return resize_1d::shape(old_op_type, *this, inputs);
        case OpType::Resize2d: return resize_2d::shape(old_op_type, *this, inputs);
        case OpType::Resize3d: return resize_3d::shape(old_op_type, *this, inputs);
        case OpType::Select: return select::shape(old_op_type, *this, inputs);
        case OpType::Sigmoid: return sigmoid::shape(old_op_type, *this, inputs);
        case OpType::Sine: return sine::shape(old_op_type, *this, inputs);
        case OpType::Softmax: return softmax::shape(old_op_type, *this, inputs);
        case OpType::SoftmaxBw: return softmax_bw::shape(old_op_type, *this, inputs);
        case OpType::SparseMatmul: return sparse_matmul::shape(old_op_type, *this, inputs);
        case OpType::Sqrt: return sqrt::shape(old_op_type, *this, inputs);
        case OpType::Squeeze: return squeeze::shape(old_op_type, *this, inputs);
        case OpType::Stack: return stack::shape(old_op_type, *this, inputs);
        case OpType::Subtract: return subtract::shape(old_op_type, *this, inputs);
        case OpType::Tanh: return tanh::shape(old_op_type, *this, inputs);
        case OpType::Tilizer: return tilizer::shape(old_op_type, *this, inputs);
        case OpType::Transpose: return transpose::shape(old_op_type, *this, inputs);
        case OpType::Unsqueeze: return unsqueeze::shape(old_op_type, *this, inputs);
        case OpType::UpdateCache: return update_cache::shape(old_op_type, *this, inputs);
        case OpType::Upsample2d: return upsample_2d::shape(old_op_type, *this, inputs);
        case OpType::Vslice: return vslice::shape(old_op_type, *this, inputs);
        case OpType::Vstack: return vstack::shape(old_op_type, *this, inputs);
        case OpType::Where: return where::shape(old_op_type, *this, inputs);
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }  // clang-format on
}

tt::graphlib::NodeContext Op::backward(
    const graphlib::OpType &old_op_type,
    tt::autograd::autograd_context &context,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    const tt::graphlib::NodeContext &output,
    const tt::graphlib::NodeContext &gradient) const
{
    switch (type_)  // clang-format off
    {
        case OpType::Abs: return abs::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::AdaptiveMaxPool2d: return adaptive_max_pool_2d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Add: return add::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::AdvIndex: return adv_index::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Argmax: return argmax::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Atan: return atan::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::AvgPool1d: return avg_pool_1d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::AvgPool2d: return avg_pool_2d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::AvgPool3d: return avg_pool_3d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Batchnorm: return batchnorm::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Broadcast: return broadcast::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Buffer: return buffer::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Cast: return cast::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Clip: return clip::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Concatenate: return concatenate::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Constant: return constant::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2d: return conv_2d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2dDepthwiseWeights: return conv_2d_depthwise_weights::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2dDepthwiseWeightsBw: return conv_2d_depthwise_weights_bw::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2dGroupedWeights: return conv_2d_grouped_weights::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2dGroupedWeightsBw: return conv_2d_grouped_weights_bw::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2dPrestrideAct: return conv_2d_prestride_act::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2dPrestrideWeights: return conv_2d_prestride_weights::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv2dTranspose: return conv_2d_transpose::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Conv3d: return conv_3d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ConvSum: return conv_sum::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Cosine: return cosine::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::CumulativeSum: return cumulative_sum::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Depthwise: return depthwise::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Dequantize: return dequantize::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Divide: return divide::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Downsample2d: return downsample_2d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::DramQueue: return dram_queue::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Dropout: return dropout::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Embedding: return embedding::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::EmbeddingBw: return embedding_bw::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Equal: return equal::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Erf: return erf::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::EthernetDatacopy: return ethernet_data_copy::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Exp: return exp::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::FillCache: return fill_cache::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ForgeDequantize: return forge_dequantize::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ForgePad: return forge_pad::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ForgeQuantize: return forge_quantize::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ForgeRequantize: return forge_requantize::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ForgeUnpad: return forge_unpad::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Gather: return gather::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Gelu: return gelu::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Greater: return greater::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::GreaterEqual: return greater_equal::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Heaviside: return heaviside::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Hslice: return hslice::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Hstack: return hstack::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Index: return index::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::IndexCopy: return index_copy::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Interleave: return interleave::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Layernorm: return layernorm::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::LayernormBw: return layernorm_bw::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::LeakyRelu: return leaky_relu::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Less: return less::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::LessEqual: return less_equal::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Log: return log::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::LogSoftmax: return log_softmax::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::LogicalAnd: return logical_and::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::LogicalNot: return logical_not::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Mask: return mask::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Matmul: return matmul::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::MaxPool1d: return max_pool_1d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::MaxPool2d: return max_pool_2d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::MaxPool3d: return max_pool_3d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Maximum: return maximum::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Minimum: return minimum::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Multiply: return multiply::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Narrow: return narrow::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Nop: return nop::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::NotEqual: return not_equal::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Pad: return pad::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::PadTile: return pad_tile::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::PixelShuffle: return pixel_shuffle::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Pow: return pow::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Power: return power::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Quantize: return quantize::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Reciprocal: return reciprocal::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ReduceAvg: return reduce_avg::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ReduceMax: return reduce_max::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::ReduceSum: return reduce_sum::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Relu: return relu::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Remainder: return remainder::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Repeat: return repeat::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::RepeatInterleave: return repeat_interleave::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Requantize: return requantize::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Reshape: return reshape::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Resize1d: return resize_1d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Resize2d: return resize_2d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Resize3d: return resize_3d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Select: return select::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Sigmoid: return sigmoid::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Sine: return sine::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Softmax: return softmax::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::SoftmaxBw: return softmax_bw::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::SparseMatmul: return sparse_matmul::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Sqrt: return sqrt::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Squeeze: return squeeze::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Stack: return stack::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Subtract: return subtract::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Tanh: return tanh::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Tilizer: return tilizer::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Transpose: return transpose::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Unsqueeze: return unsqueeze::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::UpdateCache: return update_cache::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Upsample2d: return upsample_2d::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Vslice: return vslice::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Vstack: return vstack::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        case OpType::Where: return where::backward(old_op_type, *this, context, operand, inputs, output, gradient);
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }  // clang-format on
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
    switch (type_)  // clang-format off
    {
        case OpType::Abs: return;
        case OpType::AdaptiveMaxPool2d: return adaptive_max_pool_2d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Add: return;
        case OpType::AdvIndex: return adv_index::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Argmax: return;
        case OpType::Atan: return;
        case OpType::AvgPool1d: return avg_pool_1d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::AvgPool2d: return avg_pool_2d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::AvgPool3d: return avg_pool_3d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Batchnorm: return batchnorm::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Broadcast: return broadcast::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Buffer: return buffer::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Cast: return cast::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Clip: return;
        case OpType::Concatenate: return concatenate::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Constant: return;
        case OpType::Conv2d: return conv_2d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv2dDepthwiseWeights: return conv_2d_depthwise_weights::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv2dDepthwiseWeightsBw: return conv_2d_depthwise_weights_bw::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv2dGroupedWeights: return conv_2d_grouped_weights::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv2dGroupedWeightsBw: return conv_2d_grouped_weights_bw::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv2dPrestrideAct: return conv_2d_prestride_act::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv2dPrestrideWeights: return conv_2d_prestride_weights::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv2dTranspose: return conv_2d_transpose::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Conv3d: return conv_3d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ConvSum: return conv_sum::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Cosine: return;
        case OpType::CumulativeSum: return cumulative_sum::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Depthwise: return depthwise::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Dequantize: return dequantize::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Divide: return;
        case OpType::Downsample2d: return downsample_2d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::DramQueue: return dram_queue::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Dropout: return dropout::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Embedding: return embedding::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::EmbeddingBw: return embedding_bw::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Equal: return;
        case OpType::Erf: return erf::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::EthernetDatacopy: return ethernet_data_copy::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Exp: return;
        case OpType::FillCache: return fill_cache::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ForgeDequantize: return forge_dequantize::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ForgePad: return forge_pad::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ForgeQuantize: return forge_quantize::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ForgeRequantize: return forge_requantize::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ForgeUnpad: return forge_unpad::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Gather: return gather::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Gelu: return;
        case OpType::Greater: return;
        case OpType::GreaterEqual: return;
        case OpType::Heaviside: return;
        case OpType::Hslice: return hslice::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Hstack: return hstack::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Index: return index::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::IndexCopy: return index_copy::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Interleave: return interleave::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Layernorm: return layernorm::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::LayernormBw: return layernorm_bw::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::LeakyRelu: return;
        case OpType::Less: return;
        case OpType::LessEqual: return;
        case OpType::Log: return;
        case OpType::LogSoftmax: return log_softmax::decompose_initial(old_op_type, *this, dc, inputs);;
        case OpType::LogicalAnd: return;
        case OpType::LogicalNot: return;
        case OpType::Mask: return mask::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Matmul: return matmul::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::MaxPool1d: return max_pool_1d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::MaxPool2d: return max_pool_2d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::MaxPool3d: return max_pool_3d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Maximum: return;
        case OpType::Minimum: return;
        case OpType::Multiply: return;
        case OpType::Narrow: return narrow::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Nop: return nop::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::NotEqual: return;
        case OpType::Pad: return pad::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::PadTile: return pad_tile::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::PixelShuffle: return pixel_shuffle::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Pow: return pow::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Power: return;
        case OpType::Quantize: return quantize::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Reciprocal: return;
        case OpType::ReduceAvg: return reduce_avg::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ReduceMax: return reduce_max::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::ReduceSum: return reduce_sum::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Relu: return;
        case OpType::Remainder: return;
        case OpType::Repeat: return repeat::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::RepeatInterleave: return repeat_interleave::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Requantize: return requantize::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Reshape: return reshape::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Resize1d: return resize_1d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Resize2d: return resize_2d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Resize3d: return resize_3d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Select: return select::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Sigmoid: return;
        case OpType::Sine: return;
        case OpType::Softmax: return;
        case OpType::SoftmaxBw: return;
        case OpType::SparseMatmul: return sparse_matmul::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Sqrt: return;
        case OpType::Squeeze: return;
        case OpType::Stack: return stack::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Subtract: return;
        case OpType::Tanh: return tanh::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Tilizer: return tilizer::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Transpose: return;
        case OpType::Unsqueeze: return;
        case OpType::UpdateCache: return update_cache::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Upsample2d: return upsample_2d::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Vslice: return vslice::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Vstack: return vstack::decompose_initial(old_op_type, *this, dc, inputs);
        case OpType::Where: return where::decompose_initial(old_op_type, *this, dc, inputs);
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }  // clang-format on
}

void Op::decompose_post_optimize(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const
{
    switch (type_)  // clang-format off
    {
        case OpType::Abs: return;
        case OpType::AdaptiveMaxPool2d: return adaptive_max_pool_2d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Add: return;
        case OpType::AdvIndex: return;
        case OpType::Argmax: return;
        case OpType::Atan: return;
        case OpType::AvgPool1d: return avg_pool_1d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::AvgPool2d: return avg_pool_2d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::AvgPool3d: return avg_pool_3d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Batchnorm: return batchnorm::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Broadcast: return;
        case OpType::Buffer: return buffer::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Cast: return cast::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Clip: return;
        case OpType::Concatenate: return;
        case OpType::Constant: return;
        case OpType::Conv2d: return conv_2d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv2dDepthwiseWeights: return conv_2d_depthwise_weights::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv2dDepthwiseWeightsBw: return conv_2d_depthwise_weights_bw::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv2dGroupedWeights: return conv_2d_grouped_weights::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv2dGroupedWeightsBw: return conv_2d_grouped_weights_bw::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv2dPrestrideAct: return conv_2d_prestride_act::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv2dPrestrideWeights: return conv_2d_prestride_weights::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv2dTranspose: return conv_2d_transpose::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Conv3d: return conv_3d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::ConvSum: return conv_sum::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Cosine: return;
        case OpType::CumulativeSum: return cumulative_sum::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Depthwise: return depthwise::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Dequantize: return dequantize::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Divide: return;
        case OpType::Downsample2d: return downsample_2d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::DramQueue: return dram_queue::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Dropout: return dropout::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Embedding: return embedding::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::EmbeddingBw: return embedding_bw::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Equal: return;
        case OpType::Erf: return erf::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::EthernetDatacopy: return ethernet_data_copy::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Exp: return;
        case OpType::FillCache: return fill_cache::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::ForgeDequantize: return forge_dequantize::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::ForgePad: return forge_pad::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::ForgeQuantize: return forge_quantize::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::ForgeRequantize: return forge_requantize::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::ForgeUnpad: return forge_unpad::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Gather: return gather::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Gelu: return;
        case OpType::Greater: return;
        case OpType::GreaterEqual: return;
        case OpType::Heaviside: return;
        case OpType::Hslice: return hslice::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Hstack: return hstack::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Index: return index::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::IndexCopy: return index_copy::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Interleave: return interleave::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Layernorm: return layernorm::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::LayernormBw: return layernorm_bw::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::LeakyRelu: return;
        case OpType::Less: return;
        case OpType::LessEqual: return;
        case OpType::Log: return;
        case OpType::LogSoftmax: return log_softmax::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::LogicalAnd: return;
        case OpType::LogicalNot: return;
        case OpType::Mask: return mask::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Matmul: return matmul::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::MaxPool1d: return max_pool_1d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::MaxPool2d: return max_pool_2d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::MaxPool3d: return max_pool_3d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Maximum: return;
        case OpType::Minimum: return;
        case OpType::Multiply: return;
        case OpType::Narrow: return narrow::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Nop: return nop::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::NotEqual: return;
        case OpType::Pad: return pad::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::PadTile: return pad_tile::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::PixelShuffle: return pixel_shuffle::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Pow: return pow::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Power: return;
        case OpType::Quantize: return quantize::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Reciprocal: return;
        case OpType::ReduceAvg: return;
        case OpType::ReduceMax: return;
        case OpType::ReduceSum: return;
        case OpType::Relu: return;
        case OpType::Remainder: return;
        case OpType::Repeat: return repeat::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::RepeatInterleave: return repeat_interleave::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Requantize: return requantize::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Reshape: return;
        case OpType::Resize1d: return resize_1d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Resize2d: return resize_2d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Resize3d: return resize_3d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Select: return select::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Sigmoid: return;
        case OpType::Sine: return;
        case OpType::Softmax: return;
        case OpType::SoftmaxBw: return;
        case OpType::SparseMatmul: return sparse_matmul::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Sqrt: return;
        case OpType::Squeeze: return;
        case OpType::Stack: return stack::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Subtract: return;
        case OpType::Tanh: return tanh::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Tilizer: return tilizer::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Transpose: return;
        case OpType::Unsqueeze: return;
        case OpType::UpdateCache: return update_cache::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Upsample2d: return upsample_2d::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Vslice: return vslice::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Vstack: return vstack::decompose_post_optimize(old_op_type, *this, dc, inputs);
        case OpType::Where: return where::decompose_post_optimize(old_op_type, *this, dc, inputs);
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }  // clang-format on
}

void Op::decompose_post_autograd(
    const graphlib::OpType &old_op_type,
    DecomposingContext &dc,
    const std::vector<tt::graphlib::NodeContext> &inputs) const
{
    switch (type_)  // clang-format off
    {
        case OpType::Abs: return;
        case OpType::AdaptiveMaxPool2d: return adaptive_max_pool_2d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Add: return;
        case OpType::AdvIndex: return;
        case OpType::Argmax: return;
        case OpType::Atan: return;
        case OpType::AvgPool1d: return avg_pool_1d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::AvgPool2d: return avg_pool_2d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::AvgPool3d: return avg_pool_3d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Batchnorm: return batchnorm::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Broadcast: return;
        case OpType::Buffer: return buffer::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Cast: return cast::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Clip: return;
        case OpType::Concatenate: return;
        case OpType::Constant: return;
        case OpType::Conv2d: return conv_2d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv2dDepthwiseWeights: return conv_2d_depthwise_weights::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv2dDepthwiseWeightsBw: return conv_2d_depthwise_weights_bw::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv2dGroupedWeights: return conv_2d_grouped_weights::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv2dGroupedWeightsBw: return conv_2d_grouped_weights_bw::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv2dPrestrideAct: return conv_2d_prestride_act::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv2dPrestrideWeights: return conv_2d_prestride_weights::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv2dTranspose: return conv_2d_transpose::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Conv3d: return conv_3d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::ConvSum: return conv_sum::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Cosine: return;
        case OpType::CumulativeSum: return cumulative_sum::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Depthwise: return depthwise::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Dequantize: return dequantize::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Divide: return;
        case OpType::Downsample2d: return downsample_2d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::DramQueue: return dram_queue::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Dropout: return dropout::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Embedding: return embedding::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::EmbeddingBw: return embedding_bw::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Equal: return;
        case OpType::Erf: return erf::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::EthernetDatacopy: return ethernet_data_copy::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Exp: return;
        case OpType::FillCache: return fill_cache::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::ForgeDequantize: return forge_dequantize::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::ForgePad: return forge_pad::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::ForgeQuantize: return forge_quantize::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::ForgeRequantize: return forge_requantize::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::ForgeUnpad: return forge_unpad::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Gather: return gather::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Gelu: return;
        case OpType::Greater: return;
        case OpType::GreaterEqual: return;
        case OpType::Heaviside: return heaviside::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Hslice: return hslice::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Hstack: return hstack::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Index: return index::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::IndexCopy: return index_copy::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Interleave: return interleave::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Layernorm: return layernorm::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::LayernormBw: return layernorm_bw::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::LeakyRelu: return;
        case OpType::Less: return;
        case OpType::LessEqual: return;
        case OpType::Log: return;
        case OpType::LogSoftmax: return log_softmax::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::LogicalAnd: return;
        case OpType::LogicalNot: return;
        case OpType::Mask: return mask::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Matmul: return matmul::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::MaxPool1d: return max_pool_1d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::MaxPool2d: return max_pool_2d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::MaxPool3d: return max_pool_3d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Maximum: return;
        case OpType::Minimum: return;
        case OpType::Multiply: return;
        case OpType::Narrow: return narrow::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Nop: return nop::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::NotEqual: return;
        case OpType::Pad: return pad::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::PadTile: return pad_tile::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::PixelShuffle: return pixel_shuffle::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Pow: return pow::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Power: return;
        case OpType::Quantize: return quantize::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Reciprocal: return;
        case OpType::ReduceAvg: return;
        case OpType::ReduceMax: return;
        case OpType::ReduceSum: return;
        case OpType::Relu: return;
        case OpType::Remainder: return;
        case OpType::Repeat: return repeat::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::RepeatInterleave: return repeat_interleave::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Requantize: return requantize::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Reshape: return reshape::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Resize1d: return resize_1d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Resize2d: return resize_2d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Resize3d: return resize_3d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Select: return select::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Sigmoid: return;
        case OpType::Sine: return;
        case OpType::Softmax: return;
        case OpType::SoftmaxBw: return softmax_bw::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::SparseMatmul: return sparse_matmul::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Sqrt: return;
        case OpType::Squeeze: return;
        case OpType::Stack: return stack::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Subtract: return;
        case OpType::Tanh: return tanh::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Tilizer: return tilizer::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Transpose: return;
        case OpType::Unsqueeze: return;
        case OpType::UpdateCache: return update_cache::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Upsample2d: return upsample_2d::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Vslice: return vslice::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Vstack: return vstack::decompose_post_autograd(old_op_type, *this, dc, inputs);
        case OpType::Where: return where::decompose_post_autograd(old_op_type, *this, dc, inputs);
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }  // clang-format on
}

long Op::initial_flops_estimate(
    const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    switch (type_)  // clang-format off
    {
        case OpType::Abs: return 0;
        case OpType::AdaptiveMaxPool2d: return adaptive_max_pool_2d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Add: return 0;
        case OpType::AdvIndex: return 0;
        case OpType::Argmax: return 0;
        case OpType::Atan: return 0;
        case OpType::AvgPool1d: return avg_pool_1d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::AvgPool2d: return avg_pool_2d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::AvgPool3d: return avg_pool_3d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Batchnorm: return batchnorm::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Broadcast: return 0;
        case OpType::Buffer: return buffer::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Cast: return cast::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Clip: return 0;
        case OpType::Concatenate: return 0;
        case OpType::Constant: return 0;
        case OpType::Conv2d: return conv_2d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv2dDepthwiseWeights: return conv_2d_depthwise_weights::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv2dDepthwiseWeightsBw: return conv_2d_depthwise_weights_bw::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv2dGroupedWeights: return conv_2d_grouped_weights::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv2dGroupedWeightsBw: return conv_2d_grouped_weights_bw::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv2dPrestrideAct: return conv_2d_prestride_act::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv2dPrestrideWeights: return conv_2d_prestride_weights::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv2dTranspose: return conv_2d_transpose::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Conv3d: return conv_3d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::ConvSum: return conv_sum::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Cosine: return 0;
        case OpType::CumulativeSum: return cumulative_sum::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Depthwise: return depthwise::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Dequantize: return dequantize::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Divide: return 0;
        case OpType::Downsample2d: return downsample_2d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::DramQueue: return dram_queue::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Dropout: return dropout::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Embedding: return embedding::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::EmbeddingBw: return embedding_bw::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Equal: return 0;
        case OpType::Erf: return erf::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::EthernetDatacopy: return ethernet_data_copy::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Exp: return 0;
        case OpType::FillCache: return fill_cache::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::ForgeDequantize: return forge_dequantize::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::ForgePad: return forge_pad::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::ForgeQuantize: return forge_quantize::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::ForgeRequantize: return forge_requantize::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::ForgeUnpad: return forge_unpad::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Gather: return gather::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Gelu: return 0;
        case OpType::Greater: return 0;
        case OpType::GreaterEqual: return 0;
        case OpType::Heaviside: return 0;
        case OpType::Hslice: return hslice::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Hstack: return hstack::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Index: return index::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::IndexCopy: return index_copy::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Interleave: return interleave::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Layernorm: return layernorm::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::LayernormBw: return layernorm_bw::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::LeakyRelu: return 0;
        case OpType::Less: return 0;
        case OpType::LessEqual: return 0;
        case OpType::Log: return 0;
        case OpType::LogSoftmax: return log_softmax::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::LogicalAnd: return 0;
        case OpType::LogicalNot: return 0;
        case OpType::Mask: return mask::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Matmul: return matmul::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::MaxPool1d: return max_pool_1d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::MaxPool2d: return max_pool_2d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::MaxPool3d: return max_pool_3d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Maximum: return 0;
        case OpType::Minimum: return 0;
        case OpType::Multiply: return 0;
        case OpType::Narrow: return narrow::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Nop: return nop::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::NotEqual: return 0;
        case OpType::Pad: return pad::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::PadTile: return pad_tile::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::PixelShuffle: return pixel_shuffle::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Pow: return pow::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Power: return 0;
        case OpType::Quantize: return quantize::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Reciprocal: return 0;
        case OpType::ReduceAvg: return 0;
        case OpType::ReduceMax: return 0;
        case OpType::ReduceSum: return 0;
        case OpType::Relu: return 0;
        case OpType::Remainder: return 0;
        case OpType::Repeat: return repeat::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::RepeatInterleave: return repeat_interleave::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Requantize: return requantize::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Reshape: return 0;
        case OpType::Resize1d: return resize_1d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Resize2d: return resize_2d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Resize3d: return resize_3d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Select: return select::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Sigmoid: return 0;
        case OpType::Sine: return 0;
        case OpType::Softmax: return 0;
        case OpType::SoftmaxBw: return 0;
        case OpType::SparseMatmul: return sparse_matmul::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Sqrt: return 0;
        case OpType::Squeeze: return 0;
        case OpType::Stack: return stack::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Subtract: return 0;
        case OpType::Tanh: return tanh::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Tilizer: return tilizer::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Transpose: return 0;
        case OpType::Unsqueeze: return 0;
        case OpType::UpdateCache: return update_cache::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Upsample2d: return upsample_2d::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Vslice: return vslice::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Vstack: return vstack::initial_flops_estimate(old_op_type, *this, inputs);
        case OpType::Where: return where::initial_flops_estimate(old_op_type, *this, inputs);
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }  // clang-format on
}

bool Op::is_tm(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return false;
        case OpType::AdaptiveMaxPool2d: return false;
        case OpType::Add: return false;
        case OpType::AdvIndex: return true;
        case OpType::Argmax: return false;
        case OpType::Atan: return false;
        case OpType::AvgPool1d: return false;
        case OpType::AvgPool2d: return false;
        case OpType::AvgPool3d: return false;
        case OpType::Batchnorm: return false;
        case OpType::Broadcast: return true;
        case OpType::Buffer: return false;
        case OpType::Cast: return false;
        case OpType::Clip: return false;
        case OpType::Concatenate: return false;
        case OpType::Constant: return false;
        case OpType::Conv2d: return false;
        case OpType::Conv2dDepthwiseWeights: return true;
        case OpType::Conv2dDepthwiseWeightsBw: return true;
        case OpType::Conv2dGroupedWeights: return true;
        case OpType::Conv2dGroupedWeightsBw: return true;
        case OpType::Conv2dPrestrideAct: return true;
        case OpType::Conv2dPrestrideWeights: return true;
        case OpType::Conv2dTranspose: return false;
        case OpType::Conv3d: return false;
        case OpType::ConvSum: return false;
        case OpType::Cosine: return false;
        case OpType::CumulativeSum: return false;
        case OpType::Depthwise: return false;
        case OpType::Dequantize: return false;
        case OpType::Divide: return false;
        case OpType::Downsample2d: return false;
        case OpType::DramQueue: return false;
        case OpType::Dropout: return false;
        case OpType::Embedding: return false;
        case OpType::EmbeddingBw: return false;
        case OpType::Equal: return false;
        case OpType::Erf: return false;
        case OpType::EthernetDatacopy: return false;
        case OpType::Exp: return false;
        case OpType::FillCache: return false;
        case OpType::ForgeDequantize: return false;
        case OpType::ForgePad: return true;
        case OpType::ForgeQuantize: return false;
        case OpType::ForgeRequantize: return false;
        case OpType::ForgeUnpad: return true;
        case OpType::Gather: return true;
        case OpType::Gelu: return false;
        case OpType::Greater: return false;
        case OpType::GreaterEqual: return false;
        case OpType::Heaviside: return false;
        case OpType::Hslice: return true;
        case OpType::Hstack: return true;
        case OpType::Index: return true;
        case OpType::IndexCopy: return false;
        case OpType::Interleave: return false;
        case OpType::Layernorm: return false;
        case OpType::LayernormBw: return false;
        case OpType::LeakyRelu: return false;
        case OpType::Less: return false;
        case OpType::LessEqual: return false;
        case OpType::Log: return false;
        case OpType::LogSoftmax: return false;
        case OpType::LogicalAnd: return false;
        case OpType::LogicalNot: return false;
        case OpType::Mask: return false;
        case OpType::Matmul: return false;
        case OpType::MaxPool1d: return false;
        case OpType::MaxPool2d: return false;
        case OpType::MaxPool3d: return false;
        case OpType::Maximum: return false;
        case OpType::Minimum: return false;
        case OpType::Multiply: return false;
        case OpType::Narrow: return true;
        case OpType::Nop: return false;
        case OpType::NotEqual: return false;
        case OpType::Pad: return true;
        case OpType::PadTile: return true;
        case OpType::PixelShuffle: return true;
        case OpType::Pow: return false;
        case OpType::Power: return false;
        case OpType::Quantize: return false;
        case OpType::Reciprocal: return false;
        case OpType::ReduceAvg: return false;
        case OpType::ReduceMax: return false;
        case OpType::ReduceSum: return false;
        case OpType::Relu: return false;
        case OpType::Remainder: return false;
        case OpType::Repeat: return true;
        case OpType::RepeatInterleave: return true;
        case OpType::Requantize: return false;
        case OpType::Reshape: return true;
        case OpType::Resize1d: return false;
        case OpType::Resize2d: return false;
        case OpType::Resize3d: return false;
        case OpType::Select: return true;
        case OpType::Sigmoid: return false;
        case OpType::Sine: return false;
        case OpType::Softmax: return false;
        case OpType::SoftmaxBw: return false;
        case OpType::SparseMatmul: return false;
        case OpType::Sqrt: return false;
        case OpType::Squeeze: return true;
        case OpType::Stack: return false;
        case OpType::Subtract: return false;
        case OpType::Tanh: return false;
        case OpType::Tilizer: return false;
        case OpType::Transpose: return true;
        case OpType::Unsqueeze: return true;
        case OpType::UpdateCache: return false;
        case OpType::Upsample2d: return false;
        case OpType::Vslice: return true;
        case OpType::Vstack: return true;
        case OpType::Where: return false;
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }
}

bool Op::is_eltwise(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return true;
        case OpType::AdaptiveMaxPool2d: return false;
        case OpType::Add: return true;
        case OpType::AdvIndex: return false;
        case OpType::Argmax: return true;
        case OpType::Atan: return true;
        case OpType::AvgPool1d: return false;
        case OpType::AvgPool2d: return false;
        case OpType::AvgPool3d: return false;
        case OpType::Batchnorm: return false;
        case OpType::Broadcast: return false;
        case OpType::Buffer: return true;
        case OpType::Cast: return true;
        case OpType::Clip: return true;
        case OpType::Concatenate: return true;
        case OpType::Constant: return false;
        case OpType::Conv2d: return false;
        case OpType::Conv2dDepthwiseWeights: return false;
        case OpType::Conv2dDepthwiseWeightsBw: return false;
        case OpType::Conv2dGroupedWeights: return false;
        case OpType::Conv2dGroupedWeightsBw: return false;
        case OpType::Conv2dPrestrideAct: return false;
        case OpType::Conv2dPrestrideWeights: return false;
        case OpType::Conv2dTranspose: return false;
        case OpType::Conv3d: return false;
        case OpType::ConvSum: return true;
        case OpType::Cosine: return true;
        case OpType::CumulativeSum: return true;
        case OpType::Depthwise: return false;
        case OpType::Dequantize: return false;
        case OpType::Divide: return true;
        case OpType::Downsample2d: return false;
        case OpType::DramQueue: return false;
        case OpType::Dropout: return true;
        case OpType::Embedding: return false;
        case OpType::EmbeddingBw: return false;
        case OpType::Equal: return true;
        case OpType::Erf: return true;
        case OpType::EthernetDatacopy: return true;
        case OpType::Exp: return true;
        case OpType::FillCache: return false;
        case OpType::ForgeDequantize: return false;
        case OpType::ForgePad: return false;
        case OpType::ForgeQuantize: return false;
        case OpType::ForgeRequantize: return false;
        case OpType::ForgeUnpad: return false;
        case OpType::Gather: return false;
        case OpType::Gelu: return true;
        case OpType::Greater: return true;
        case OpType::GreaterEqual: return true;
        case OpType::Heaviside: return true;
        case OpType::Hslice: return false;
        case OpType::Hstack: return false;
        case OpType::Index: return false;
        case OpType::IndexCopy: return true;
        case OpType::Interleave: return true;
        case OpType::Layernorm: return false;
        case OpType::LayernormBw: return false;
        case OpType::LeakyRelu: return true;
        case OpType::Less: return true;
        case OpType::LessEqual: return true;
        case OpType::Log: return true;
        case OpType::LogSoftmax: return false;
        case OpType::LogicalAnd: return true;
        case OpType::LogicalNot: return true;
        case OpType::Mask: return false;
        case OpType::Matmul: return false;
        case OpType::MaxPool1d: return false;
        case OpType::MaxPool2d: return false;
        case OpType::MaxPool3d: return false;
        case OpType::Maximum: return true;
        case OpType::Minimum: return true;
        case OpType::Multiply: return true;
        case OpType::Narrow: return false;
        case OpType::Nop: return true;
        case OpType::NotEqual: return true;
        case OpType::Pad: return false;
        case OpType::PadTile: return false;
        case OpType::PixelShuffle: return false;
        case OpType::Pow: return true;
        case OpType::Power: return true;
        case OpType::Quantize: return false;
        case OpType::Reciprocal: return true;
        case OpType::ReduceAvg: return false;
        case OpType::ReduceMax: return false;
        case OpType::ReduceSum: return false;
        case OpType::Relu: return true;
        case OpType::Remainder: return true;
        case OpType::Repeat: return false;
        case OpType::RepeatInterleave: return false;
        case OpType::Requantize: return false;
        case OpType::Reshape: return false;
        case OpType::Resize1d: return false;
        case OpType::Resize2d: return false;
        case OpType::Resize3d: return false;
        case OpType::Select: return false;
        case OpType::Sigmoid: return true;
        case OpType::Sine: return true;
        case OpType::Softmax: return false;
        case OpType::SoftmaxBw: return false;
        case OpType::SparseMatmul: return false;
        case OpType::Sqrt: return true;
        case OpType::Squeeze: return false;
        case OpType::Stack: return true;
        case OpType::Subtract: return true;
        case OpType::Tanh: return true;
        case OpType::Tilizer: return true;
        case OpType::Transpose: return false;
        case OpType::Unsqueeze: return false;
        case OpType::UpdateCache: return false;
        case OpType::Upsample2d: return false;
        case OpType::Vslice: return false;
        case OpType::Vstack: return false;
        case OpType::Where: return true;
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }
}

bool Op::is_eltwise_unary(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return true;
        case OpType::AdaptiveMaxPool2d: return false;
        case OpType::Add: return false;
        case OpType::AdvIndex: return false;
        case OpType::Argmax: return true;
        case OpType::Atan: return true;
        case OpType::AvgPool1d: return false;
        case OpType::AvgPool2d: return false;
        case OpType::AvgPool3d: return false;
        case OpType::Batchnorm: return false;
        case OpType::Broadcast: return false;
        case OpType::Buffer: return true;
        case OpType::Cast: return true;
        case OpType::Clip: return true;
        case OpType::Concatenate: return false;
        case OpType::Constant: return false;
        case OpType::Conv2d: return false;
        case OpType::Conv2dDepthwiseWeights: return false;
        case OpType::Conv2dDepthwiseWeightsBw: return false;
        case OpType::Conv2dGroupedWeights: return false;
        case OpType::Conv2dGroupedWeightsBw: return false;
        case OpType::Conv2dPrestrideAct: return false;
        case OpType::Conv2dPrestrideWeights: return false;
        case OpType::Conv2dTranspose: return false;
        case OpType::Conv3d: return false;
        case OpType::ConvSum: return false;
        case OpType::Cosine: return true;
        case OpType::CumulativeSum: return true;
        case OpType::Depthwise: return false;
        case OpType::Dequantize: return false;
        case OpType::Divide: return false;
        case OpType::Downsample2d: return false;
        case OpType::DramQueue: return false;
        case OpType::Dropout: return true;
        case OpType::Embedding: return false;
        case OpType::EmbeddingBw: return false;
        case OpType::Equal: return false;
        case OpType::Erf: return true;
        case OpType::EthernetDatacopy: return true;
        case OpType::Exp: return true;
        case OpType::FillCache: return false;
        case OpType::ForgeDequantize: return false;
        case OpType::ForgePad: return false;
        case OpType::ForgeQuantize: return false;
        case OpType::ForgeRequantize: return false;
        case OpType::ForgeUnpad: return false;
        case OpType::Gather: return false;
        case OpType::Gelu: return true;
        case OpType::Greater: return false;
        case OpType::GreaterEqual: return false;
        case OpType::Heaviside: return false;
        case OpType::Hslice: return false;
        case OpType::Hstack: return false;
        case OpType::Index: return false;
        case OpType::IndexCopy: return false;
        case OpType::Interleave: return false;
        case OpType::Layernorm: return false;
        case OpType::LayernormBw: return false;
        case OpType::LeakyRelu: return true;
        case OpType::Less: return false;
        case OpType::LessEqual: return false;
        case OpType::Log: return true;
        case OpType::LogSoftmax: return false;
        case OpType::LogicalAnd: return false;
        case OpType::LogicalNot: return true;
        case OpType::Mask: return false;
        case OpType::Matmul: return false;
        case OpType::MaxPool1d: return false;
        case OpType::MaxPool2d: return false;
        case OpType::MaxPool3d: return false;
        case OpType::Maximum: return false;
        case OpType::Minimum: return false;
        case OpType::Multiply: return false;
        case OpType::Narrow: return false;
        case OpType::Nop: return true;
        case OpType::NotEqual: return false;
        case OpType::Pad: return false;
        case OpType::PadTile: return false;
        case OpType::PixelShuffle: return false;
        case OpType::Pow: return true;
        case OpType::Power: return false;
        case OpType::Quantize: return false;
        case OpType::Reciprocal: return true;
        case OpType::ReduceAvg: return false;
        case OpType::ReduceMax: return false;
        case OpType::ReduceSum: return false;
        case OpType::Relu: return true;
        case OpType::Remainder: return false;
        case OpType::Repeat: return false;
        case OpType::RepeatInterleave: return false;
        case OpType::Requantize: return false;
        case OpType::Reshape: return false;
        case OpType::Resize1d: return false;
        case OpType::Resize2d: return false;
        case OpType::Resize3d: return false;
        case OpType::Select: return false;
        case OpType::Sigmoid: return true;
        case OpType::Sine: return true;
        case OpType::Softmax: return false;
        case OpType::SoftmaxBw: return false;
        case OpType::SparseMatmul: return false;
        case OpType::Sqrt: return true;
        case OpType::Squeeze: return false;
        case OpType::Stack: return false;
        case OpType::Subtract: return false;
        case OpType::Tanh: return true;
        case OpType::Tilizer: return true;
        case OpType::Transpose: return false;
        case OpType::Unsqueeze: return false;
        case OpType::UpdateCache: return false;
        case OpType::Upsample2d: return false;
        case OpType::Vslice: return false;
        case OpType::Vstack: return false;
        case OpType::Where: return false;
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }
}

bool Op::is_eltwise_binary(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return false;
        case OpType::AdaptiveMaxPool2d: return false;
        case OpType::Add: return true;
        case OpType::AdvIndex: return false;
        case OpType::Argmax: return false;
        case OpType::Atan: return false;
        case OpType::AvgPool1d: return false;
        case OpType::AvgPool2d: return false;
        case OpType::AvgPool3d: return false;
        case OpType::Batchnorm: return false;
        case OpType::Broadcast: return false;
        case OpType::Buffer: return false;
        case OpType::Cast: return false;
        case OpType::Clip: return false;
        case OpType::Concatenate: return false;
        case OpType::Constant: return false;
        case OpType::Conv2d: return false;
        case OpType::Conv2dDepthwiseWeights: return false;
        case OpType::Conv2dDepthwiseWeightsBw: return false;
        case OpType::Conv2dGroupedWeights: return false;
        case OpType::Conv2dGroupedWeightsBw: return false;
        case OpType::Conv2dPrestrideAct: return false;
        case OpType::Conv2dPrestrideWeights: return false;
        case OpType::Conv2dTranspose: return false;
        case OpType::Conv3d: return false;
        case OpType::ConvSum: return false;
        case OpType::Cosine: return false;
        case OpType::CumulativeSum: return false;
        case OpType::Depthwise: return false;
        case OpType::Dequantize: return false;
        case OpType::Divide: return true;
        case OpType::Downsample2d: return false;
        case OpType::DramQueue: return false;
        case OpType::Dropout: return false;
        case OpType::Embedding: return false;
        case OpType::EmbeddingBw: return false;
        case OpType::Equal: return true;
        case OpType::Erf: return false;
        case OpType::EthernetDatacopy: return false;
        case OpType::Exp: return false;
        case OpType::FillCache: return false;
        case OpType::ForgeDequantize: return false;
        case OpType::ForgePad: return false;
        case OpType::ForgeQuantize: return false;
        case OpType::ForgeRequantize: return false;
        case OpType::ForgeUnpad: return false;
        case OpType::Gather: return false;
        case OpType::Gelu: return false;
        case OpType::Greater: return true;
        case OpType::GreaterEqual: return true;
        case OpType::Heaviside: return true;
        case OpType::Hslice: return false;
        case OpType::Hstack: return false;
        case OpType::Index: return false;
        case OpType::IndexCopy: return false;
        case OpType::Interleave: return false;
        case OpType::Layernorm: return false;
        case OpType::LayernormBw: return false;
        case OpType::LeakyRelu: return false;
        case OpType::Less: return true;
        case OpType::LessEqual: return true;
        case OpType::Log: return false;
        case OpType::LogSoftmax: return false;
        case OpType::LogicalAnd: return true;
        case OpType::LogicalNot: return false;
        case OpType::Mask: return false;
        case OpType::Matmul: return false;
        case OpType::MaxPool1d: return false;
        case OpType::MaxPool2d: return false;
        case OpType::MaxPool3d: return false;
        case OpType::Maximum: return true;
        case OpType::Minimum: return true;
        case OpType::Multiply: return true;
        case OpType::Narrow: return false;
        case OpType::Nop: return false;
        case OpType::NotEqual: return true;
        case OpType::Pad: return false;
        case OpType::PadTile: return false;
        case OpType::PixelShuffle: return false;
        case OpType::Pow: return false;
        case OpType::Power: return true;
        case OpType::Quantize: return false;
        case OpType::Reciprocal: return false;
        case OpType::ReduceAvg: return false;
        case OpType::ReduceMax: return false;
        case OpType::ReduceSum: return false;
        case OpType::Relu: return false;
        case OpType::Remainder: return true;
        case OpType::Repeat: return false;
        case OpType::RepeatInterleave: return false;
        case OpType::Requantize: return false;
        case OpType::Reshape: return false;
        case OpType::Resize1d: return false;
        case OpType::Resize2d: return false;
        case OpType::Resize3d: return false;
        case OpType::Select: return false;
        case OpType::Sigmoid: return false;
        case OpType::Sine: return false;
        case OpType::Softmax: return false;
        case OpType::SoftmaxBw: return false;
        case OpType::SparseMatmul: return false;
        case OpType::Sqrt: return false;
        case OpType::Squeeze: return false;
        case OpType::Stack: return false;
        case OpType::Subtract: return true;
        case OpType::Tanh: return false;
        case OpType::Tilizer: return false;
        case OpType::Transpose: return false;
        case OpType::Unsqueeze: return false;
        case OpType::UpdateCache: return false;
        case OpType::Upsample2d: return false;
        case OpType::Vslice: return false;
        case OpType::Vstack: return false;
        case OpType::Where: return false;
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
    }
}

bool Op::is_eltwise_nary(const graphlib::OpType &old_op_type) const
{
    switch (type_)
    {
        case OpType::Abs: return false;
        case OpType::AdaptiveMaxPool2d: return false;
        case OpType::Add: return false;
        case OpType::AdvIndex: return false;
        case OpType::Argmax: return false;
        case OpType::Atan: return false;
        case OpType::AvgPool1d: return false;
        case OpType::AvgPool2d: return false;
        case OpType::AvgPool3d: return false;
        case OpType::Batchnorm: return false;
        case OpType::Broadcast: return false;
        case OpType::Buffer: return false;
        case OpType::Cast: return false;
        case OpType::Clip: return false;
        case OpType::Concatenate: return true;
        case OpType::Constant: return false;
        case OpType::Conv2d: return false;
        case OpType::Conv2dDepthwiseWeights: return false;
        case OpType::Conv2dDepthwiseWeightsBw: return false;
        case OpType::Conv2dGroupedWeights: return false;
        case OpType::Conv2dGroupedWeightsBw: return false;
        case OpType::Conv2dPrestrideAct: return false;
        case OpType::Conv2dPrestrideWeights: return false;
        case OpType::Conv2dTranspose: return false;
        case OpType::Conv3d: return false;
        case OpType::ConvSum: return true;
        case OpType::Cosine: return false;
        case OpType::CumulativeSum: return false;
        case OpType::Depthwise: return false;
        case OpType::Dequantize: return false;
        case OpType::Divide: return false;
        case OpType::Downsample2d: return false;
        case OpType::DramQueue: return false;
        case OpType::Dropout: return false;
        case OpType::Embedding: return false;
        case OpType::EmbeddingBw: return false;
        case OpType::Equal: return false;
        case OpType::Erf: return false;
        case OpType::EthernetDatacopy: return false;
        case OpType::Exp: return false;
        case OpType::FillCache: return false;
        case OpType::ForgeDequantize: return false;
        case OpType::ForgePad: return false;
        case OpType::ForgeQuantize: return false;
        case OpType::ForgeRequantize: return false;
        case OpType::ForgeUnpad: return false;
        case OpType::Gather: return false;
        case OpType::Gelu: return false;
        case OpType::Greater: return false;
        case OpType::GreaterEqual: return false;
        case OpType::Heaviside: return false;
        case OpType::Hslice: return false;
        case OpType::Hstack: return false;
        case OpType::Index: return false;
        case OpType::IndexCopy: return true;
        case OpType::Interleave: return true;
        case OpType::Layernorm: return false;
        case OpType::LayernormBw: return false;
        case OpType::LeakyRelu: return false;
        case OpType::Less: return false;
        case OpType::LessEqual: return false;
        case OpType::Log: return false;
        case OpType::LogSoftmax: return false;
        case OpType::LogicalAnd: return false;
        case OpType::LogicalNot: return false;
        case OpType::Mask: return false;
        case OpType::Matmul: return false;
        case OpType::MaxPool1d: return false;
        case OpType::MaxPool2d: return false;
        case OpType::MaxPool3d: return false;
        case OpType::Maximum: return false;
        case OpType::Minimum: return false;
        case OpType::Multiply: return false;
        case OpType::Narrow: return false;
        case OpType::Nop: return false;
        case OpType::NotEqual: return false;
        case OpType::Pad: return false;
        case OpType::PadTile: return false;
        case OpType::PixelShuffle: return false;
        case OpType::Pow: return false;
        case OpType::Power: return false;
        case OpType::Quantize: return false;
        case OpType::Reciprocal: return false;
        case OpType::ReduceAvg: return false;
        case OpType::ReduceMax: return false;
        case OpType::ReduceSum: return false;
        case OpType::Relu: return false;
        case OpType::Remainder: return false;
        case OpType::Repeat: return false;
        case OpType::RepeatInterleave: return false;
        case OpType::Requantize: return false;
        case OpType::Reshape: return false;
        case OpType::Resize1d: return false;
        case OpType::Resize2d: return false;
        case OpType::Resize3d: return false;
        case OpType::Select: return false;
        case OpType::Sigmoid: return false;
        case OpType::Sine: return false;
        case OpType::Softmax: return false;
        case OpType::SoftmaxBw: return false;
        case OpType::SparseMatmul: return false;
        case OpType::Sqrt: return false;
        case OpType::Squeeze: return false;
        case OpType::Stack: return true;
        case OpType::Subtract: return true;
        case OpType::Tanh: return false;
        case OpType::Tilizer: return false;
        case OpType::Transpose: return false;
        case OpType::Unsqueeze: return false;
        case OpType::UpdateCache: return false;
        case OpType::Upsample2d: return false;
        case OpType::Vslice: return false;
        case OpType::Vstack: return false;
        case OpType::Where: return true;
        default: TT_ASSERT(false, "Unknown OpType."); unreachable();
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
