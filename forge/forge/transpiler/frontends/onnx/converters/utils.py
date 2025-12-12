"""
ONNX-specific utility functions.
"""
import onnx
from onnx import ModelProto
from typing import List, Tuple


def remove_initializers_from_input(model: ModelProto) -> ModelProto:
    """
    Removes initializer names from the graph's input list. 
    """
    graph_inputs = model.graph.input
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    
    # Filter out inputs that are also initializers. Modifies list in place.
    for i in range(len(graph_inputs) - 1, -1, -1):
        if graph_inputs[i].name in initializer_names:
            del graph_inputs[i]

    return model


def get_inputs_names(onnx_graph) -> List[str]:
    """Get list of input names (excluding initializers)."""
    param_names = {x.name for x in onnx_graph.initializer}
    input_names = [x.name for x in onnx_graph.input if x.name not in param_names]
    return input_names


def get_outputs_names(onnx_graph) -> List[str]:
    """Get list of output names."""
    return [x.name for x in onnx_graph.output]


def compute_autopad_padding(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    mode: str = "SAME_UPPER"
) -> Tuple[int, int]:
    """
    Compute padding values for a single dimension based on ONNX auto_pad modes.
    
    This is a utility function for computing padding values based on ONNX's auto_pad
    attribute. The padding values are then used to create PadNode instances in the graph.
    
    Args:
        input_size: Input size in this dimension
        kernel_size: Kernel size in this dimension
        stride: Stride in this dimension
        dilation: Dilation in this dimension
        mode: "SAME_UPPER", "SAME_LOWER", or "VALID"
        
    Returns:
        Tuple of (pad_before, pad_after) for this dimension
        
    Example:
        For Conv2d with input H=32, kernel=3, stride=1, dilation=1, mode="SAME_UPPER":
        >>> pad_before, pad_after = compute_autopad_padding(32, 3, 1, 1, "SAME_UPPER")
        >>> # Returns (1, 1) - pad 1 pixel before and after
    """
    if mode == "VALID":
        return (0, 0)
    
    # Calculate effective kernel size with dilation
    # Effective kernel = (kernel_size - 1) * dilation + 1
    # Example: kernel=3, dilation=2 -> effective = (3-1)*2 + 1 = 5
    effective_kernel = (kernel_size - 1) * dilation + 1
    
    # Calculate output size (ceil division)
    # output_size = ceil(input_size / stride)
    output_size = (input_size + stride - 1) // stride
    
    # Total padding needed to achieve the desired output size
    # Formula: total_pad = (output_size - 1) * stride + effective_kernel - input_size
    total_pad = max(0, (output_size - 1) * stride + effective_kernel - input_size)
    
    if mode == "SAME_UPPER":
        # More padding on the right/bottom (end of dimension)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
    else:  # SAME_LOWER
        # More padding on the left/top (beginning of dimension)
        pad_after = total_pad // 2
        pad_before = total_pad - pad_after
    
    return (pad_before, pad_after)

