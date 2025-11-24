"""
ONNX-specific utility functions.
"""
import onnx
from onnx import ModelProto
from typing import List


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

