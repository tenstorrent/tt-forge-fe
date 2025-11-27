"""
Shape/Reshape operations: Flatten, Reshape, Transpose, Squeeze, Unsqueeze
"""
import torch
from typing import Dict, List

from ..nodes import TIRNode
from ..types import TensorInfo



class ReshapeNode(TIRNode):
    """
    PyTorch-like Reshape operation.
    Takes one input tensor and shape as attribute (matching torch.reshape API).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               shape: tuple) -> 'ReshapeNode':
        """Static factory method to create a ReshapeNode."""
        return ReshapeNode(
            name=name,
            op_type="Reshape",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'shape': shape},
            forge_op_function_name="forge.op.tm.Reshape"
        )
    
    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        shape = self.attrs.get('shape', None)
        
        if shape is None:
            # Fallback: use output shape from TensorInfo
            output_info = list(self.output_tensors.values())[0]
            if output_info and output_info.shape:
                shape = tuple(s if s is not None else x.shape[i] 
                            for i, s in enumerate(output_info.shape))
            else:
                shape = x.shape
        
        # Handle 0 in shape (ONNX convention: 0 means keep that dimension from input)
        if isinstance(shape, (list, tuple)):
            shape = list(shape)
            for i, s in enumerate(shape):
                if s == 0:
                    shape[i] = x.shape[i]
            shape = tuple(shape)
        
        return {self.outputs[0]: torch.reshape(x, shape)}



class TransposeNode(TIRNode):
    """
    PyTorch-like Transpose operation that swaps two dimensions.
    For multi-dimensional transpositions, create multiple TransposeNode instances.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim0: int, dim1: int) -> 'TransposeNode':
        """Static factory method to create a TransposeNode."""
        return TransposeNode(
            name=name,
            op_type="Transpose",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim0': dim0, 'dim1': dim1},
            forge_op_function_name="forge.op.tm.Transpose"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        dim0 = self.attrs['dim0']
        dim1 = self.attrs['dim1']
        return {self.outputs[0]: torch.transpose(x, dim0, dim1)}



class SqueezeNode(TIRNode):
    """
    PyTorch-like Squeeze operation.
    Takes one input tensor and dim as attribute (matching torch.squeeze API).
    Forge Squeeze requires dim as attribute (single int).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: int) -> 'SqueezeNode':
        """Static factory method to create a SqueezeNode."""
        return SqueezeNode(
            name=name,
            op_type="Squeeze",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim},
            forge_op_function_name="forge.op.tm.Squeeze"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs. Forge Squeeze requires dim as int (not tuple)."""
        forge_attrs = {}
        if 'dim' in attrs:
            dim = attrs['dim']
            if isinstance(dim, (list, tuple)):
                # If multiple dims, use first one (Forge limitation)
                forge_attrs['dim'] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs['dim'] = dim
        return forge_attrs

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        dim = self.attrs.get('dim', None)
        if dim is not None:
            # Forge Squeeze only supports single dim, so if we have multiple dims,
            # we need to handle it (but Forge API only supports one)
            if isinstance(dim, (list, tuple)):
                # Squeeze multiple dims in reverse order to avoid index shifting
                result = x
                for d in sorted(dim, reverse=True):
                    result = torch.squeeze(result, dim=d)
                return {self.outputs[0]: result}
            else:
                return {self.outputs[0]: torch.squeeze(x, dim=dim)}
        else:
            return {self.outputs[0]: torch.squeeze(x)}

