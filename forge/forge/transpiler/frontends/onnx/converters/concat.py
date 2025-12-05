"""
ONNX Concat operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.other import ConcatNode
from .base import OnnxOpConverter


class ConcatConverter(OnnxOpConverter):
    """Converter for ONNX Concat operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Concat opset v1-v3: axis as attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Concat_{node_index}"
        axis = attrs.get('axis', 0)
        
        return [ConcatNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]
    
    @classmethod
    def _impl_v4(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Concat opset v4+: axis as attribute (same as v1, but with additional validation).
        """
        # Same implementation as v1
        return cls._impl_v1(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Concat opset v11+: axis as attribute (same as v4, but with better type handling).
        """
        # Same implementation as v1/v4
        return cls._impl_v1(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)

