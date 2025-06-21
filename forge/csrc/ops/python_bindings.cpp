// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ops/python_bindings.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <memory>

#include "autograd/autograd.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "ops/op.hpp"
#include "passes/decomposing_context.hpp"
#include "torch/extension.h"

namespace tt
{

// template <ops::OpType type, class OpAttrs>
// std::unique_ptr<ops::Op> make_op(OpAttrs &&attrs)
// {
// return std::make_unique<ops::Op>(type, std::forward(attrs));
// }

void OpsModule(py::module &m_ops)
{
    py::enum_<ops::OpType>(m_ops, "OpType")
        .value("Abs", ops::OpType::Abs)
        .value("AdaptiveMaxPool2d", ops::OpType::AdaptiveMaxPool2d)
        .value("Add", ops::OpType::Add)
        .value("AdvIndex", ops::OpType::AdvIndex)
        .value("Argmax", ops::OpType::Argmax)
        .value("Atan", ops::OpType::Atan)
        .value("AvgPool1d", ops::OpType::AvgPool1d)
        .value("AvgPool2d", ops::OpType::AvgPool2d)
        .value("AvgPool3d", ops::OpType::AvgPool3d)
        .value("Batchnorm", ops::OpType::Batchnorm)
        .value("Broadcast", ops::OpType::Broadcast)
        .value("Buffer", ops::OpType::Buffer)
        .value("Cast", ops::OpType::Cast)
        .value("Clip", ops::OpType::Clip)
        .value("Concatenate", ops::OpType::Concatenate)
        .value("Constant", ops::OpType::Constant)
        .value("Conv2d", ops::OpType::Conv2d)
        .value("Conv2dDepthwiseWeights", ops::OpType::Conv2dDepthwiseWeights)
        .value("Conv2dDepthwiseWeightsBw", ops::OpType::Conv2dDepthwiseWeightsBw)
        .value("Conv2dGroupedWeights", ops::OpType::Conv2dGroupedWeights)
        .value("Conv2dGroupedWeightsBw", ops::OpType::Conv2dGroupedWeightsBw)
        .value("Conv2dPrestrideAct", ops::OpType::Conv2dPrestrideAct)
        .value("Conv2dPrestrideWeights", ops::OpType::Conv2dPrestrideWeights)
        .value("Conv2dTranspose", ops::OpType::Conv2dTranspose)
        .value("Conv3d", ops::OpType::Conv3d)
        .value("ConvSum", ops::OpType::ConvSum)
        .value("Cosine", ops::OpType::Cosine)
        .value("CumulativeSum", ops::OpType::CumulativeSum)
        .value("Dequantize", ops::OpType::Dequantize)
        .value("Depthwise", ops::OpType::Depthwise)
        .value("Divide", ops::OpType::Divide)
        .value("DramQueue", ops::OpType::DramQueue)
        .value("Dropout", ops::OpType::Dropout)
        .value("Embedding", ops::OpType::Embedding)
        .value("EmbeddingBw", ops::OpType::EmbeddingBw)
        .value("Equal", ops::OpType::Equal)
        .value("Erf", ops::OpType::Erf)
        .value("EthernetDatacopy", ops::OpType::EthernetDatacopy)
        .value("Exp", ops::OpType::Exp)
        .value("ForgeDequantize", ops::OpType::ForgeDequantize)
        .value("ForgePad", ops::OpType::ForgePad)
        .value("ForgeQuantize", ops::OpType::ForgeQuantize)
        .value("ForgeRequantize", ops::OpType::ForgeRequantize)
        .value("ForgeUnpad", ops::OpType::ForgeUnpad)
        .value("Gather", ops::OpType::Gather)
        .value("Gelu", ops::OpType::Gelu)
        .value("GeluDerivative", ops::OpType::GeluDerivative)
        .value("Greater", ops::OpType::Greater)
        .value("GreaterEqual", ops::OpType::GreaterEqual)
        .value("GroupedReduceAvg", ops::OpType::GroupedReduceAvg)
        .value("Heaviside", ops::OpType::Heaviside)
        .value("Hslice", ops::OpType::Hslice)
        .value("Hstack", ops::OpType::Hstack)
        .value("Index", ops::OpType::Index)
        .value("IndexCopy", ops::OpType::IndexCopy)
        .value("Interleave", ops::OpType::Interleave)
        .value("Layernorm", ops::OpType::Layernorm)
        .value("LayernormBw", ops::OpType::LayernormBw)
        .value("LeakyRelu", ops::OpType::LeakyRelu)
        .value("Less", ops::OpType::Less)
        .value("LessEqual", ops::OpType::LessEqual)
        .value("Log", ops::OpType::Log)
        .value("LogSoftmax", ops::OpType::LogSoftmax)
        .value("LogicalAnd", ops::OpType::LogicalAnd)
        .value("LogicalNot", ops::OpType::LogicalNot)
        .value("Mask", ops::OpType::Mask)
        .value("Matmul", ops::OpType::Matmul)
        .value("Maximum", ops::OpType::Maximum)
        .value("Minimum", ops::OpType::Minimum)
        .value("Multiply", ops::OpType::Multiply)
        .value("Nop", ops::OpType::Nop)
        .value("NotEqual", ops::OpType::NotEqual)
        .value("Narrow", ops::OpType::Narrow)
        .value("Pad", ops::OpType::Pad)
        .value("PadTile", ops::OpType::PadTile)
        .value("PixelShuffle", ops::OpType::PixelShuffle)
        .value("Pow", ops::OpType::Pow)
        .value("Power", ops::OpType::Power)
        .value("Quantize", ops::OpType::Quantize)
        .value("Reciprocal", ops::OpType::Reciprocal)
        .value("ReduceAvg", ops::OpType::ReduceAvg)
        .value("ReduceMax", ops::OpType::ReduceMax)
        .value("ReduceSum", ops::OpType::ReduceSum)
        .value("Relu", ops::OpType::Relu)
        .value("Remainder", ops::OpType::Remainder)
        .value("Repeat", ops::OpType::Repeat)
        .value("RepeatInterleave", ops::OpType::RepeatInterleave)
        .value("Requantize", ops::OpType::Requantize)
        .value("Reshape", ops::OpType::Reshape)
        .value("Resize1d", ops::OpType::Resize1d)
        .value("Resize2d", ops::OpType::Resize2d)
        .value("Resize3d", ops::OpType::Resize3d)
        .value("Select", ops::OpType::Select)
        .value("Sigmoid", ops::OpType::Sigmoid)
        .value("Sine", ops::OpType::Sine)
        .value("Softmax", ops::OpType::Softmax)
        .value("SoftmaxBw", ops::OpType::SoftmaxBw)
        .value("SparseMatmul", ops::OpType::SparseMatmul)
        .value("Sqrt", ops::OpType::Sqrt)
        .value("Stack", ops::OpType::Stack)
        .value("Subtract", ops::OpType::Subtract)
        .value("Squeeze", ops::OpType::Squeeze)
        .value("Tanh", ops::OpType::Tanh)
        .value("TileBroadcast", ops::OpType::TileBroadcast)
        .value("Tilizer", ops::OpType::Tilizer)
        .value("Transpose", ops::OpType::Transpose)
        .value("Unsqueeze", ops::OpType::Unsqueeze)
        .value("Upsample2d", ops::OpType::Upsample2d)
        .value("Vslice", ops::OpType::Vslice)
        .value("Vstack", ops::OpType::Vstack)
        .value("Where", ops::OpType::Where)
        .export_values();

    // Check whether this needs to change to py::class_<Op, std::unique_ptr<Op>>(m_ops, "Op").
    py::class_<ops::Op>(m_ops, "Op")
        .def(
            py::init([](ops::OpType type, ops::Attrs attrs)
                     { return std::make_unique<ops::Op>(type, std::move(attrs)); }),
            py::arg("type") = ops::OpType{},
            py::arg("attr") = ops::Attrs{})
        .def("attr", [](const ops::Op &self, const std::string &name) { return self.attr(name); })
        .def("type", [](const ops::Op &self) { return self.type(); })
        .def("eval", &ops::Op::eval)
        .def("shape", &ops::Op::shape)
        .def("backward", &ops::Op::backward)
        .def("decompose", &ops::Op::decompose)
        .def("initial_flops_stimate", &ops::Op::initial_flops_estimate);

    py::class_<ops::OpAbs, ops::Op>(m_ops, "OpAbs")
        .def(
            py::init([](ops::Attrs attrs) { return std::make_unique<ops::OpAbs>(std::move(attrs)); }),
            py::arg("attr") = ops::Attrs{});
}

}  // namespace tt
