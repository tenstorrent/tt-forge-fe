"""
Intermediate Representation (IR) module.

Contains common IR types and operations shared across all frontends.
"""
from .types import TensorInfo, onnx_dtype_to_torch_dtype
from .nodes import TIRNode

# Import operations
from .operations import *

__all__ = [
    'TensorInfo',
    'onnx_dtype_to_torch_dtype',
    'TIRNode',
]

