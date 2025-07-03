// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace at
{
class Tensor;  // Forward declaration of torch tensor.
}

namespace tt
{
namespace graphlib
{
struct OpType;
class Shape;
struct NodeContext;
using DimBroadcastTrampoline = std::tuple<int, int, int>;
}  // namespace graphlib

namespace autograd
{
struct autograd_context;
}

class DecomposingContext;
enum class DecomposeEpoch : uint8_t;

namespace ops
{

enum class OpType : uint32_t
{
    Abs,
    AdaptiveMaxPool2d,
    Add,
    AdvIndex,
    Argmax,
    Atan,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    Batchnorm,
    Broadcast,
    Buffer,
    Cast,
    Clip,
    Concatenate,
    Constant,
    Conv2d,
    Conv2dDepthwiseWeights,
    Conv2dDepthwiseWeightsBw,
    Conv2dGroupedWeights,
    Conv2dGroupedWeightsBw,
    Conv2dPrestrideAct,
    Conv2dPrestrideWeights,
    Conv2dTranspose,
    Conv3d,
    ConvSum,
    Cosine,
    CumulativeSum,
    Dequantize,
    Depthwise,
    Divide,
    DramQueue,
    Downsample2d,
    Dropout,
    Embedding,
    EmbeddingBw,
    Equal,
    Erf,
    EthernetDatacopy,
    Exp,
    FillCache,
    ForgeDequantize,
    ForgePad,
    ForgeQuantize,
    ForgeRequantize,
    ForgeUnpad,
    Gather,
    Gelu,
    GeluDerivative,
    Greater,
    GreaterEqual,
    GroupedReduceAvg,
    Heaviside,
    Hslice,
    Hstack,
    Index,
    IndexCopy,
    Interleave,
    Layernorm,
    LayernormBw,
    LeakyRelu,
    Less,
    LessEqual,
    Log,
    LogSoftmax,
    LogicalAnd,
    LogicalNot,
    Mask,
    Matmul,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    Maximum,
    Minimum,
    Multiply,
    Nop,
    NotEqual,
    Narrow,
    Pad,
    PadTile,
    PixelShuffle,
    Pow,
    Power,
    Quantize,
    Reciprocal,
    ReduceAvg,
    ReduceMax,
    ReduceSum,
    Relu,
    Remainder,
    Repeat,
    RepeatInterleave,
    Requantize,
    Reshape,
    Resize1d,
    Resize2d,
    Resize3d,
    Select,
    Sigmoid,
    Sine,
    Softmax,
    SoftmaxBw,
    SparseMatmul,
    Sqrt,
    Stack,
    Subtract,
    Squeeze,
    Tanh,
    TileBroadcast,
    Tilizer,
    Transpose,
    Unsqueeze,
    UpdateCache,
    Upsample2d,
    Vslice,
    Vstack,
    Where,
};

/**
 * Op attribute.
 */
using Attr = ::std::variant<
    std::string,
    bool,
    int,
    float,
    std::vector<int>,
    std::vector<std::tuple<int, int, int>>,
    std::vector<std::tuple<int, int, int, int>>>;

/**
 * Op attributes.
 * TODO: Conver this into vector. There is no sence to use map/unordered map for few items. Also, once migrated to new
 * ops, we should create attribute key enum, and replace mapping strings.
 */
using Attrs = ::std::map<std::string, Attr>;

class Op
{
   public:
    Op(OpType type, Attrs attrs) : type_(type), attrs_(std::move(attrs)) {}
    Op(OpType type) : type_(type) {}
    Op(const graphlib::OpType &old_op_type);

    bool operator==(const Op &other) const { return type_ == other.type_ && attrs_ == other.attrs_; }
    bool operator!=(const Op &other) const { return !(*this == other); }

    OpType type() const { return type_; }
    const Attrs &attrs() const { return attrs_; }

    const Attr &attr(std::string const &name) const { return attrs_.at(name); }
    template <typename T>
    const T &attr_as(std::string const &name) const
    {
        return std::get<T>(attr(name));
    }

    bool has_attr(const std::string &attr_name) const { return attrs_.find(attr_name) != attrs_.end(); }
    void set_attrs(Attrs attrs) { attrs_ = std::move(attrs); }
    void set_attr(std::string const &name, Attr attr) { attrs_[name] = attr; }

    const std::string &as_string() const;

    /* ----------------------------------------------------*
     * Calculations segment. All ops must implement these. *
     * ----------------------------------------------------*/

    at::Tensor eval(const graphlib::OpType &old_op_type, const std::vector<at::Tensor> &tensors) const;

    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> shape(
        const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const;

    tt::graphlib::NodeContext backward(
        const graphlib::OpType &old_op_type,
        tt::autograd::autograd_context &context,
        int operand,
        const std::vector<tt::graphlib::NodeContext> &inputs,
        const tt::graphlib::NodeContext &output,
        const tt::graphlib::NodeContext &gradient) const;

    bool is_tm(const graphlib::OpType &old_op_type) const;
    bool is_eltwise(const graphlib::OpType &old_op_type) const;
    bool is_eltwise_unary(const graphlib::OpType &old_op_type) const;
    bool is_eltwise_binary(const graphlib::OpType &old_op_type) const;
    bool is_eltwise_nary(const graphlib::OpType &old_op_type) const;

    /* --------------------------*
     * Optional implementations. *
     * --------------------------*/

    /**
     * Note: We will most likely get rid of distinct implementations for decompose, once we investigate why they even
     * exist. They are needed for now in order to unblock ops migration from python to cpp.
     */
    template <DecomposeEpoch epoch>
    void decompose(
        const graphlib::OpType &old_op_type,
        DecomposingContext &dc,
        const std::vector<tt::graphlib::NodeContext> &inputs) const;

    void decompose_initial(
        const graphlib::OpType &old_op_type,
        DecomposingContext &dc,
        const std::vector<tt::graphlib::NodeContext> &inputs) const;

    void decompose_post_optimize(
        const graphlib::OpType &old_op_type,
        DecomposingContext &dc,
        const std::vector<tt::graphlib::NodeContext> &inputs) const;

    void decompose_post_autograd(
        const graphlib::OpType &old_op_type,
        DecomposingContext &dc,
        const std::vector<tt::graphlib::NodeContext> &inputs) const;

    long initial_flops_estimate(
        const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const;

   private:
    /* ------------------------------------------------------------*
     * Base - common for all ops that are not yet migrated to cpp. *
     * ------------------------------------------------------------*/

    at::Tensor base_eval(const graphlib::OpType &old_op_type, const std::vector<at::Tensor> &tensors) const;

    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcastTrampoline>> base_shape(
        const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const;

    tt::graphlib::NodeContext base_backward(
        const graphlib::OpType &old_op_type,
        tt::autograd::autograd_context &context,
        int operand,
        const std::vector<tt::graphlib::NodeContext> &inputs,
        const tt::graphlib::NodeContext &output,
        const tt::graphlib::NodeContext &gradient) const;

    bool base_is_tm(const graphlib::OpType &old_op_type) const;
    bool base_is_eltwise(const graphlib::OpType &old_op_type) const;
    bool base_is_eltwise_unary(const graphlib::OpType &old_op_type) const;
    bool base_is_eltwise_binary(const graphlib::OpType &old_op_type) const;
    bool base_is_eltwise_nary(const graphlib::OpType &old_op_type) const;

    void base_decompose(
        const graphlib::OpType &old_op_type,
        const char *dispatch,
        DecomposingContext &dc,
        const std::vector<tt::graphlib::NodeContext> &inputs) const;

    long base_initial_flops_estimate(
        const graphlib::OpType &old_op_type, const std::vector<std::vector<std::uint32_t>> &inputs) const;

   private:
    OpType type_;
    Attrs attrs_;
};

}  // namespace ops
}  // namespace tt
