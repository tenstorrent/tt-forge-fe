// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/TensorBody.h>

#include <string>
#include <tuple>
#include <unordered_map>
#include <utils/assert.hpp>
#include <variant>
#include <vector>

#include "autograd/autograd.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"

namespace tt
{
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
    Dropout,
    Embedding,
    EmbeddingBw,
    Equal,
    Erf,
    EthernetDatacopy,
    Exp,
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
    Upsample2d,
    Vslice,
    Vstack,
    Where,
};

// Op attribute.
using Attr = ::std::variant<
    std::string,
    bool,
    int,
    float,
    std::vector<int>,
    std::vector<std::tuple<int, int, int>>,
    std::vector<std::tuple<int, int, int, int>>>;

// Op attributes.
using Attrs = ::std::unordered_map<std::string, Attr>;

using Shape = graphlib::Shape;

class Op
{
   public:
    Op(OpType type, Attrs attrs) : type_(type), attrs_(std::move(attrs)) {}
    Op(OpType type) : type_(type) {}

    virtual ~Op() = default;

    bool operator==(const Op &other) const { return type_ == other.type_ && attrs_ == other.attrs_; }
    bool operator!=(const Op &other) const { return !(*this == other); }

    OpType type() const { return type_; }

    Attr const &attr(std::string const &name) const { return attrs_.at(name); }
    Attr &attr(std::string const &name) { return attrs_.at(name); }
    template <typename T>
    T const &attr_as(std::string const &name) const
    {
        return std::get<T>(attr(name));
    }
    template <typename T>
    T &attr_as(std::string const &name)
    {
        return std::get<T>(attr(name));
    }

    void set_attr(std::string const &name, Attr attr) { attrs_[name] = attr; }

    graphlib::OpType as_old_op_type() const;

    // ========================================
    // Calculations. Derived classes must implement these.
    // ========================================

    virtual at::Tensor eval(const std::vector<at::Tensor> &tensors) const;
    virtual Shape shape(const std::vector<std::vector<std::uint32_t>> &inputs) const;
    virtual long initial_flops_stimate(const std::vector<std::vector<std::uint32_t>> &inputs) const;

    virtual tt::graphlib::NodeContext backward(
        tt::autograd::autograd_context context,
        int operand,
        const std::vector<tt::graphlib::NodeContext> &inputs,
        tt::graphlib::NodeContext output,
        tt::graphlib::NodeContext gradient) const;

   private:
    OpType type_;
    Attrs attrs_;
};

class OpAbs : public Op
{
   public:
    OpAbs(Attrs attrs) : Op(OpType::Abs, std::move(attrs)) {}

    at::Tensor eval(const std::vector<at::Tensor> &tensors) const override;
    Shape shape(const std::vector<std::vector<std::uint32_t>> &in_shapes) const override;
    long initial_flops_stimate(const std::vector<std::vector<std::uint32_t>> &inputs) const override;
};

class OpAdd : public Op
{
   public:
    OpAdd(Attrs attrs) : Op(OpType::Add, std::move(attrs)) {}

    at::Tensor eval(const std::vector<at::Tensor> &tensors) const override;
    Shape shape(const std::vector<std::vector<std::uint32_t>> &in_shapes) const override;
    long initial_flops_stimate(const std::vector<std::vector<std::uint32_t>> &inputs) const override;
};

}  // namespace ops
}  // namespace tt
