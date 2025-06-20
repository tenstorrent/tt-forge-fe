// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

namespace at
{
class Tensor;
}

namespace tt
{
namespace graphlib
{
struct OpType;
class Shape;
struct NodeContext;
}  // namespace graphlib

namespace autograd
{
struct autograd_context;
}

class DecomposingContext;

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

class Op
{
   public:
    Op(OpType type, Attrs attrs) : type_(type), attrs_(std::move(attrs)) {}
    Op(OpType type) : type_(type) {}

    virtual ~Op() = default;

    bool operator==(const Op &other) const { return type_ == other.type_ && attrs_ == other.attrs_; }
    bool operator!=(const Op &other) const { return !(*this == other); }

    OpType type() const { return type_; }
    const Attrs &attrs() const { return attrs_; }

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

    bool has_attr(const std::string &attr_name) const { return attrs_.find(attr_name) != attrs_.end(); }
    void set_attrs(Attrs attrs) { attrs_ = std::move(attrs); }
    void set_attr(std::string const &name, Attr attr) { attrs_[name] = attr; }

    graphlib::OpType as_old_op_type() const;

    // ========================================
    // Calculations. Derived classes must implement these.
    // ========================================

    virtual at::Tensor eval(const std::vector<at::Tensor> &tensors) const;
    virtual graphlib::Shape shape(const std::vector<std::vector<std::uint32_t>> &inputs) const;

    virtual tt::graphlib::NodeContext backward(
        tt::autograd::autograd_context context,
        int operand,
        const std::vector<tt::graphlib::NodeContext> &inputs,
        tt::graphlib::NodeContext output,
        tt::graphlib::NodeContext gradient) const;

    virtual void decompose(
        const char *dispatch, DecomposingContext &dc, std::vector<tt::graphlib::NodeContext> &inputs) const;

    virtual long initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const;

    virtual bool is_tm() const;
    virtual bool is_eltwise() const;
    virtual bool is_eltwise_unary() const;
    virtual bool is_eltwise_binary() const;
    virtual bool is_eltwise_nary() const;

   private:
    OpType type_;
    Attrs attrs_;
};

class OpTM : public Op
{
    using Op::Op;
    bool is_tm() const override { return true; };
    bool is_eltwise() const override { return false; };
    bool is_eltwise_unary() const override { return false; };
    bool is_eltwise_binary() const override { return false; };
    bool is_eltwise_nary() const override { return false; };
};

class OpEltwise : public Op
{
    using Op::Op;
    bool is_tm() const override { return false; };
    bool is_eltwise() const override { return true; };
    bool is_eltwise_unary() const override { return false; };
    bool is_eltwise_binary() const override { return false; };
    bool is_eltwise_nary() const override { return false; };
};

class OpEltwiseUnary : public OpEltwise
{
    using OpEltwise::OpEltwise;
    bool is_eltwise_unary() const override { return true; };
};

class OpEltwiseBinary : public OpEltwise
{
    using OpEltwise::OpEltwise;
    bool is_eltwise_binary() const override { return true; };
};

class OpEltwiseNary : public OpEltwise
{
    using OpEltwise::OpEltwise;
    bool is_eltwise_nary() const override { return true; };
};

// Check whether this is needed once refactoring is done.
// It seems that user can always create wanted op where it is needed.
std::unique_ptr<Op> create_op(OpType op_type);

///////////////////////////////////////////////////
// Next section contains ops implemented in cpp. //
///////////////////////////////////////////////////

class OpAbs : public OpEltwiseUnary
{
   public:
    OpAbs() : OpEltwiseUnary(OpType::Abs) {}
    explicit OpAbs(Attrs attrs) : OpEltwiseUnary(OpType::Abs, std::move(attrs)) {}

    at::Tensor eval(const std::vector<at::Tensor> &tensors) const override;
    graphlib::Shape shape(const std::vector<std::vector<std::uint32_t>> &in_shapes) const override;
    long initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const override;
};

class OpAdd : public OpEltwiseBinary
{
   public:
    OpAdd() : OpEltwiseBinary(OpType::Add) {}
    explicit OpAdd(Attrs attrs) : OpEltwiseBinary(OpType::Add, std::move(attrs)) {}

    at::Tensor eval(const std::vector<at::Tensor> &tensors) const override;
    graphlib::Shape shape(const std::vector<std::vector<std::uint32_t>> &in_shapes) const override;
    long initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const override;
};

}  // namespace ops
}  // namespace tt
