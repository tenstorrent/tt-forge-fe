"""
Forge Transpiler Package

A multi-frontend transpiler for converting ML framework models to Forge intermediate representation.
Supports ONNX, with PaddlePaddle and TensorFlow coming soon.
"""

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import operations to register them (must be imported first)
from .ir.operations import *

# Public API - IR (common across all frontends)
from .ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from .ir.nodes import TIRNode

# Public API - Core
from .core.graph import TIRGraph

# Public API - Code Generation
from .codegen import generate_forge_module

# Public API - ONNX Frontend
from .frontends.onnx import ONNXToForgeTranspiler
from .frontends.onnx.converters import (
    extract_attributes,
    extract_attr_value,
    AutoPad,
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
)
from .frontends.onnx.debug import debug_node_output, get_activation_value

# Public API - Common Utils
from .utils import (
    is_constant,
    is_symmetric_padding,
    extract_padding_for_conv,
    get_selection,
)

__all__ = [
    # Types (IR)
    'TensorInfo',
    'onnx_dtype_to_torch_dtype',
    # Nodes (IR)
    'TIRNode',
    # Graph (Core)
    'TIRGraph',
    # Code Generation
    'generate_forge_module',
    # ONNX Frontend
    'ONNXToForgeTranspiler',
    # ONNX Converters
    'extract_attributes',
    'extract_attr_value',
    'AutoPad',
    'remove_initializers_from_input',
    'get_inputs_names',
    'get_outputs_names',
    # ONNX Debug
    'debug_node_output',
    'get_activation_value',
    # Common Utils
    'is_constant',
    'is_symmetric_padding',
    'extract_padding_for_conv',
    'get_selection',
]
