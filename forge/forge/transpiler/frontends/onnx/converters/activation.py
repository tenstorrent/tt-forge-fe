"""
ONNX Activation operation converters (Relu, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyRelu).
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.activation import (
    ReluNode, SigmoidNode, TanhNode, SoftmaxNode, LogSoftmaxNode, LeakyReluNode
)
from .base import OnnxOpConverter


class ReluConverter(OnnxOpConverter):
    """Converter for ONNX Relu operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Relu opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"Relu_{node_index}"
        return [ReluNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class SigmoidConverter(OnnxOpConverter):
    """Converter for ONNX Sigmoid operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Sigmoid opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"Sigmoid_{node_index}"
        return [SigmoidNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class TanhConverter(OnnxOpConverter):
    """Converter for ONNX Tanh operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Tanh opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"Tanh_{node_index}"
        return [TanhNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class SoftmaxConverter(OnnxOpConverter):
    """Converter for ONNX Softmax operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Softmax opset v1-v10: axis defaults to 1."""
        node_name = node_proto.name if node_proto.name else f"Softmax_{node_index}"
        axis = attrs.get('axis', 1)
        return [SoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """Softmax opset v11+: axis defaults to -1."""
        node_name = node_proto.name if node_proto.name else f"Softmax_{node_index}"
        axis = attrs.get('axis', -1)
        return [SoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]


class LogSoftmaxConverter(OnnxOpConverter):
    """Converter for ONNX LogSoftmax operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """LogSoftmax opset v1-v10: axis defaults to 1."""
        node_name = node_proto.name if node_proto.name else f"LogSoftmax_{node_index}"
        axis = attrs.get('axis', 1)
        return [LogSoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """LogSoftmax opset v11+: axis defaults to -1."""
        node_name = node_proto.name if node_proto.name else f"LogSoftmax_{node_index}"
        axis = attrs.get('axis', -1)
        return [LogSoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]


class LeakyReluConverter(OnnxOpConverter):
    """Converter for ONNX LeakyRelu operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """LeakyRelu opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"LeakyRelu_{node_index}"
        alpha = attrs.get('alpha', 0.01)
        return [LeakyReluNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            negative_slope=alpha
        )]

