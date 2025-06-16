// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "op.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <unordered_map>
#include <utils/logger.hpp>

#include "graph_lib/node_types.hpp"
#include "lower_to_forge/common.hpp"
#include "torch/extension.h"
#include "torch/torch.h"

namespace tt
{
namespace ops
{

namespace py = pybind11;

// In transition period we need mapping from new to the old node type, in order to preserve old functionalities.
class OldOpMapper
{
   public:
    OldOpMapper()
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
        mapping_[OpType::Dequantize] = "dequantize";
        mapping_[OpType::Depthwise] = "depthwise";
        mapping_[OpType::Divide] = "divide";
        mapping_[OpType::DramQueue] = "dram_queue";
        mapping_[OpType::Dropout] = "dropout";
        mapping_[OpType::Embedding] = "embedding";
        mapping_[OpType::EmbeddingBw] = "embedding_bw";
        mapping_[OpType::Equal] = "equal";
        mapping_[OpType::Erf] = "erf";
        mapping_[OpType::EthernetDatacopy] = "ethernet_datacopy";
        mapping_[OpType::Exp] = "exp";
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
        mapping_[OpType::Maximum] = "maximum";
        mapping_[OpType::Minimum] = "minimum";
        mapping_[OpType::Multiply] = "multiply";
        mapping_[OpType::Nop] = "nop";
        mapping_[OpType::NotEqual] = "not_equal";
        mapping_[OpType::Narrow] = "narrow";
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
        mapping_[OpType::Stack] = "stack";
        mapping_[OpType::Subtract] = "subtract";
        mapping_[OpType::Squeeze] = "squeeze";
        mapping_[OpType::Tanh] = "tanh";
        mapping_[OpType::TileBroadcast] = "tile_broadcast";
        mapping_[OpType::Tilizer] = "tilizer";
        mapping_[OpType::Transpose] = "transpose";
        mapping_[OpType::Unsqueeze] = "unsqueeze";
        mapping_[OpType::Upsample2d] = "upsample2d";
        mapping_[OpType::Vslice] = "vslice";
        mapping_[OpType::Vstack] = "vstack";
        mapping_[OpType::Where] = "where";
    }

    const std::string &operator[](OpType op_type) const { return mapping_.at(op_type); }

   private:
    std::unordered_map<OpType, std::string> mapping_;
};

static OldOpMapper mapper;

// Constructs old attributes from provided new attributes. Since new Attr is superset of the old ForgeOpAttrs,
// we can just visit new attribute.
ForgeOpAttrs as_old_attrs(const Attrs &attrs)
{
    ForgeOpAttrs ret_attrs;
    for (const auto &attr : attrs)
        ret_attrs[attr.first] = std::visit([](const auto &value) -> ForgeOpAttr { return value; }, attr.second);

    return ret_attrs;
}

graphlib::OpType Op::as_old_op_type() const { return graphlib::OpType(mapper[type_], {}, {}, as_old_attrs(attrs_)); }

// py::object eval_op(graphlib::OpType type, std::vector<py::object> inputs, bool evaluate_output_relu = true)
// {
// py::object eval_module = py::module_::import("forge.op.eval.forge");
// py::function forge_eval = eval_module.attr("get_f_forge_eval")(std::ref(type));

// log_trace(LogEval, "  eval_op: {}", type);
// bool has_requant =
// type.forge_attrs.find("requant") != type.forge_attrs.end() and std::get<bool>(type.forge_attrs.at("requant"));

// std::vector<py::object> inputs_;
// if (has_requant)
// {
// inputs_.assign(inputs.begin(), inputs.end());
// inputs_.erase(inputs_.end() - 1);  // skip requantization input (last input)
// }
// else
// {
// inputs_ = inputs;
// }

// py::object result = forge_eval(inputs_);

// py::object common_module = py::module_::import("forge.op.eval");
// common_module.attr("eval_debug_print")(type.op, inputs, result);

// if (evaluate_output_relu)
// result = eval_relu(result, type);

// return result;
// }

// Default implementation for ops that are not cpp implemented yet. We will invoke old python code to evaluate them.
torch::Tensor Op::eval(const std::vector<torch::Tensor> &tensors) const
{
    graphlib::OpType old_op_type = as_old_op_type();

    py::object eval_module = py::module_::import("forge.op.eval.forge");
    py::function forge_eval = eval_module.attr("get_f_forge_eval")(std::ref(old_op_type));

    py::object result = forge_eval(tensors);
    return result.cast<torch::Tensor>();
}

Shape Op::shape(const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    graphlib::OpType old_op_type = as_old_op_type();

    py::object eval_module = py::module_::import("forge.op.eval.forge");
    py::function forge_shape = eval_module.attr("get_f_forge_shape")(std::ref(old_op_type));

    py::object result = forge_shape(inputs);

    // Old code returns tuple of (shape, [])
    auto tuple_result = result.cast<py::tuple>();
    if (tuple_result.size() != 2)
        throw std::runtime_error("Expected a tuple of shape and broadcast.");

    Shape shape = Shape::create(tuple_result[0].cast<std::vector<std::uint32_t>>());
    return shape;
}

long Op::initial_flops_stimate(const std::vector<std::vector<std::uint32_t>> &inputs) const
{
    graphlib::OpType old_op_type = as_old_op_type();

    py::object eval_module = py::module_::import("forge.op.eval.forge");
    py::function initial_flops_estimate = eval_module.attr("get_f_forge_initial_flops_estimate")(old_op_type);
    py::object ret = initial_flops_estimate(inputs);

    return ret.cast<long>();
}

tt::graphlib::NodeContext Op::backward(
    tt::autograd::autograd_context context,
    int operand,
    const std::vector<tt::graphlib::NodeContext> &inputs,
    tt::graphlib::NodeContext output,
    tt::graphlib::NodeContext gradient) const
{
    graphlib::OpType old_op_type = as_old_op_type();

    auto eval_module = py::module_::import("forge.op.eval.forge");
    py::function forge_backward = eval_module.attr("get_f_forge_backward")(old_op_type);

    return forge_backward(context, operand, inputs, output, gradient).cast<tt::graphlib::NodeContext>();
}

}  // namespace ops

}  // namespace tt
