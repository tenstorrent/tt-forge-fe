"""
Enhanced attribute extraction for ONNX nodes.
"""
import warnings
import onnx
from onnx import numpy_helper
from typing import Dict, Any


def extract_attr_value(attr) -> Any:
    """Extract a single attribute value from ONNX attribute."""
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8')
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8') for s in attr.strings]
    elif attr.type == onnx.AttributeProto.TENSORS:
        return [numpy_helper.to_array(t) for t in attr.tensors]
    elif attr.type == onnx.AttributeProto.GRAPH:
        return attr.g
    elif attr.type == onnx.AttributeProto.GRAPHS:
        return list(attr.graphs)
    else:
        warnings.warn(f"Unsupported attribute type: {attr.type}")
        return None


def extract_attributes(node) -> Dict[str, Any]:
    """
    Extract attributes from an ONNX node with proper name mapping.
    Maps ONNX attribute names to PyTorch/Forge-friendly names.
    """
    attrs = {}
    
    for attr in node.attribute:
        attr_name = attr.name
        attr_value = extract_attr_value(attr)
        
        if attr_value is None:
            continue
            
        # Map ONNX attribute names to PyTorch/Forge names
        if attr_name == "axis" and node.op_type == "Flatten":
            attrs["start_dim"] = attr_value
        elif attr_name == "axis" or attr_name == "axes":
            # Convert single-element list to scalar
            if isinstance(attr_value, list) and len(attr_value) == 1:
                attrs["dim"] = attr_value[0]
            else:
                attrs["dim"] = attr_value
        elif attr_name == "dilations":
            attrs["dilation"] = tuple(attr_value) if isinstance(attr_value, list) else attr_value
        elif attr_name == "kernel_shape":
            attrs["kernel_size"] = tuple(attr_value) if isinstance(attr_value, list) else attr_value
        elif attr_name == "strides":
            attrs["stride"] = tuple(attr_value) if isinstance(attr_value, list) else attr_value
        elif attr_name == "pads":
            attrs["pads"] = attr_value  # Will be processed by auto_pad logic
        elif attr_name == "auto_pad":
            attrs["auto_pad"] = attr_value.decode('utf-8') if isinstance(attr_value, bytes) else attr_value
        elif attr_name == "group":
            attrs["groups"] = attr_value
        elif attr_name == "keepdims":
            attrs["keepdim"] = bool(attr_value)
        elif attr_name == "epsilon":
            attrs["eps"] = attr_value
        elif attr_name == "momentum":
            attrs["momentum"] = attr_value
        elif attr_name == "alpha":
            attrs["alpha"] = attr_value
        elif attr_name == "beta":
            attrs["beta"] = attr_value
        elif attr_name == "transA":
            attrs["transA"] = bool(attr_value)
        elif attr_name == "transB":
            attrs["transB"] = bool(attr_value)
        elif attr_name == "to":
            # Cast operation - map ONNX dtype to string
            if isinstance(attr_value, int):
                dtype_map = {
                    onnx.TensorProto.FLOAT: "float32",
                    onnx.TensorProto.DOUBLE: "float64",
                    onnx.TensorProto.INT32: "int32",
                    onnx.TensorProto.INT64: "int64",
                    onnx.TensorProto.BOOL: "bool",
                }
                attrs["to"] = dtype_map.get(attr_value, "float32")
            else:
                attrs["to"] = attr_value
        elif attr_name == "perm":
            attrs["perm"] = tuple(attr_value) if isinstance(attr_value, list) else attr_value
        elif attr_name == "split":
            attrs["split"] = attr_value
        elif attr_name == "ceil_mode":
            attrs["ceil_mode"] = bool(attr_value)
        elif attr_name == "min":
            attrs["min"] = attr_value
        elif attr_name == "max":
            attrs["max"] = attr_value
        elif attr_name == "value":
            attrs["value"] = attr_value
        elif attr_name == "mode":
            attrs["mode"] = attr_value.decode('utf-8') if isinstance(attr_value, bytes) else attr_value
        else:
            # For unknown attributes, use original name
            attrs[attr_name] = attr_value
    
    return attrs

