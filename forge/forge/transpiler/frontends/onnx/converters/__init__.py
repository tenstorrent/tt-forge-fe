"""
ONNX conversion utilities.
"""
from .attributes import extract_attributes, extract_attr_value, AttributeParsingError
from .utils import (
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
    torch_dtype_to_onnx_dtype,
)
from .validation import (
    ValidationError,
    validate_attributes,
    validate_constant_input,
    handle_validation_error,
)

__all__ = [
    'extract_attributes',
    'extract_attr_value',
    'AttributeParsingError',
    'remove_initializers_from_input',
    'get_inputs_names',
    'get_outputs_names',
    'torch_dtype_to_onnx_dtype',
    'ValidationError',
    'validate_attributes',
    'validate_constant_input',
    'handle_validation_error',
]

