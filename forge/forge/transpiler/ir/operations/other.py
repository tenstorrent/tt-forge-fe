"""
Other operations: Concat, Clip, Cast, Pad, Identity
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

from ..nodes import TIRNode
from ..types import TensorInfo, onnx_dtype_to_torch_dtype



class ConcatNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dim: int = 0) -> 'ConcatNode':
        """Static factory method to create a ConcatNode."""
        return ConcatNode(
            name=name,
            op_type="Concat",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'dim': dim},
            forge_op_function_name="forge.op.Concatenate"
        )
    
    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch dim to Forge axis."""
        if 'dim' in attrs:
            return {'axis': attrs['dim']}
        return {}

    def eval(self, input_tensors):
        inputs = [input_tensors[inp] for inp in self.inputs]
        dim = self.attrs.get('dim', 0)
        return {self.outputs[0]: torch.cat(inputs, dim=dim)}



class ClipNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               min_val: float = None,
               max_val: float = None) -> 'ClipNode':
        """Static factory method to create a ClipNode."""
        attrs = {}
        if min_val is not None:
            attrs['min'] = min_val
        if max_val is not None:
            attrs['max'] = max_val
        return ClipNode(
            name=name,
            op_type="Clip",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
            forge_op_function_name="forge.op.Clip"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        min_val = self.attrs.get('min', None)
        max_val = self.attrs.get('max', None)
        
        # Handle min/max from inputs if provided
        if len(self.inputs) > 1:
            min_val = input_tensors.get(self.inputs[1], min_val)
        if len(self.inputs) > 2:
            max_val = input_tensors.get(self.inputs[2], max_val)
            
        return {self.outputs[0]: torch.clamp(x, min=min_val, max=max_val)}



class CastNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               dtype: Optional[torch.dtype] = None) -> 'CastNode':
        """Static factory method to create a CastNode."""
        attrs = {}
        if dtype is not None:
            attrs['dtype'] = dtype
        return CastNode(
            name=name,
            op_type="Cast",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
            forge_op_function_name="forge.op.Cast"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        to_dtype = self.attrs.get('dtype', x.dtype)
        return {self.outputs[0]: x.to(dtype=to_dtype)}



class PadNode(TIRNode):
    """
    PyTorch-like Pad operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               pad: Union[List[int], Tuple[int, ...]],
               mode: str = "constant",
               value: float = 0.0) -> 'PadNode':
        """Static factory method to create a PadNode."""
        return PadNode(
            name=name,
            op_type="Pad",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'pad': pad, 'mode': mode, 'value': value},
            forge_op_function_name="forge.op.misc.Pad"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        pad = self.attrs['pad']
        mode = self.attrs.get('mode', 'constant')
        value = self.attrs.get('value', 0.0)
        return {self.outputs[0]: F.pad(x, pad, mode=mode, value=value)}



class IdentityNode(TIRNode):
    """
    Identity operation - returns input tensor unchanged.
    Maps to PyTorch's identity operation (just returns the tensor as-is).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'IdentityNode':
        """Static factory method to create an IdentityNode."""
        return IdentityNode(
            name=name,
            op_type="Identity",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Identity"
        )

    def eval(self, input_tensors):
        """
        Evaluate Identity operation.
        Returns the input tensor unchanged (identity function).
        """
        x = input_tensors[self.inputs[0]]
        # Return the tensor as-is (identity operation)
        return {self.outputs[0]: x}

