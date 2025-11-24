"""
Generic fallback node for unsupported operations.
"""
import torch
import logging
from typing import Dict, List, Any

from ..nodes import TIRNode
from ..types import TensorInfo

logger = logging.getLogger("ForgeTranspiler")


class GenericNode(TIRNode):
    """Generic fallback node for unsupported operations."""
    def __init__(self, 
                 name: str, 
                 op_type: str,
                 inputs: List[str], 
                 outputs: List[str],
                 input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo],
                 attrs: Dict[str, Any] = None,
                 forge_op_function_name: str = None):
        """Initialize GenericNode with PyTorch-compatible attributes."""
        if attrs is None:
            attrs = {}
        if forge_op_function_name is None:
            forge_op_function_name = f"forge.op.{op_type.lower()}"
        super().__init__(
            name=name,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
            forge_op_function_name=forge_op_function_name
        )
    
    def eval(self, input_tensors):
        logger.warning(f"Eval not implemented for {self.op_type}. Returning zero tensor.")
        # Use TensorInfo to get the shape for the dummy output
        dummy_outputs = {}
        for name, info in self.output_tensors.items():
            if info.shape is not None:
                # Replace None (dynamic) dims with a placeholder, e.g., 1
                shape_tuple = tuple(d if isinstance(d, int) else 1 for d in info.shape)
                dummy_outputs[name] = torch.zeros(shape_tuple, dtype=info.torch_dtype)
        return dummy_outputs

