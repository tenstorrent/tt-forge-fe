"""
Node naming utilities for ONNX to TIR conversion.
Provides consistent naming conventions for TIR nodes.
"""
import re
from typing import Optional
import logging

logger = logging.getLogger("ForgeTranspiler")


def sanitize_name(name: str) -> str:
    """
    Sanitize a node name to be Python-identifier friendly.
    
    Replaces invalid characters with underscores and ensures the name
    doesn't start with a digit.
    
    Args:
        name: Original node name
        
    Returns:
        Sanitized name safe for use as Python identifier
    """
    if not name:
        return name
    
    # Replace invalid characters with underscores
    # Invalid: ., /, :, -, spaces, and other special chars
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure name doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f'node_{sanitized}'
    
    # Ensure name is not empty
    if not sanitized:
        sanitized = 'unnamed_node'
    
    return sanitized


def generate_node_name(op_type: str, 
                       node_index: int,
                       onnx_name: Optional[str] = None,
                       prefix: Optional[str] = None) -> str:
    """
    Generate a consistent node name for TIR nodes.
    
    Priority:
    1. Use ONNX node name if provided (sanitized)
    2. Use prefix + op_type + index if prefix provided
    3. Use op_type + index as fallback
    
    Args:
        op_type: Operation type (e.g., "Conv", "Add")
        node_index: Index of the node in the graph
        onnx_name: Original ONNX node name (optional)
        prefix: Optional prefix for generated names
        
    Returns:
        Generated node name
    """
    # If ONNX provides a name, use it (after sanitization)
    if onnx_name:
        sanitized = sanitize_name(onnx_name)
        if sanitized:
            return sanitized
    
    # Generate name based on op_type and index
    op_type_clean = op_type.replace('/', '_').replace(':', '_')
    
    if prefix:
        name = f"{prefix}_{op_type_clean}_{node_index}"
    else:
        name = f"{op_type_clean}_{node_index}"
    
    return sanitize_name(name)


def ensure_unique_name(name: str, existing_names: set, counter: int = 0) -> str:
    """
    Ensure a node name is unique by appending a counter if needed.
    
    Args:
        name: Proposed node name
        existing_names: Set of existing node names
        counter: Starting counter value
        
    Returns:
        Unique node name
    """
    if name not in existing_names:
        return name
    
    # Try appending counter
    base_name = name
    while f"{base_name}_{counter}" in existing_names:
        counter += 1
    
    unique_name = f"{base_name}_{counter}"
    return unique_name

