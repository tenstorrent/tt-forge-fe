"""
Arithmetic operations: Add, Sub, Mul, Div, MatMul, Gemm
"""
import torch
from typing import Dict, List

from ..nodes import TIRNode
from ..types import TensorInfo
from ...utils.helpers import is_constant


class AddNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'AddNode':
        """Static factory method to create an AddNode."""
        return AddNode(
            name=name,
            op_type="Add",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Add"
        )
    
    def eval(self, input_tensors):
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        
        # Use utility function to check if inputs are constants
        # Reorder inputs so that constants come last for better broadcasting
        if is_constant(a):
            a, b = b, a
        
        return {self.outputs[0]: torch.add(a, b)}



class SubNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'SubNode':
        """Static factory method to create a SubNode."""
        return SubNode(
            name=name,
            op_type="Sub",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Subtract"
        )
    
    def eval(self, input_tensors):
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        return {self.outputs[0]: torch.sub(a, b)}



class MulNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'MulNode':
        """Static factory method to create a MulNode."""
        return MulNode(
            name=name,
            op_type="Mul",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Multiply"
        )
    
    def eval(self, input_tensors):
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        return {self.outputs[0]: torch.mul(a, b)}



class DivNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'DivNode':
        """Static factory method to create a DivNode."""
        return DivNode(
            name=name,
            op_type="Div",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Divide"
        )
    
    def eval(self, input_tensors):
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        return {self.outputs[0]: torch.div(a, b)}



class MatMulNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'MatMulNode':
        """Static factory method to create a MatMulNode."""
        return MatMulNode(
            name=name,
            op_type="MatMul",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Matmul"
        )
    
    def eval(self, input_tensors):
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        return {self.outputs[0]: torch.matmul(a, b)}


