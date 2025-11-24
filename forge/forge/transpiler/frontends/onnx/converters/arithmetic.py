"""
ONNX Arithmetic operation converters (Add, Sub, Mul, Div, MatMul).
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.arithmetic import AddNode, SubNode, MulNode, DivNode, MatMulNode
from .base import OnnxOpConverter


class AddConverter(OnnxOpConverter):
    """Converter for ONNX Add operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Add opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"Add_{node_index}"
        return [AddNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class SubConverter(OnnxOpConverter):
    """Converter for ONNX Sub operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Sub opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"Sub_{node_index}"
        return [SubNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class MulConverter(OnnxOpConverter):
    """Converter for ONNX Mul operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Mul opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"Mul_{node_index}"
        return [MulNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class DivConverter(OnnxOpConverter):
    """Converter for ONNX Div operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Div opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"Div_{node_index}"
        return [DivNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class MatMulConverter(OnnxOpConverter):
    """Converter for ONNX MatMul operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """MatMul opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"MatMul_{node_index}"
        return [MatMulNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]

