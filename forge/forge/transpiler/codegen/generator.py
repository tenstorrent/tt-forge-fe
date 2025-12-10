"""
Code generation for converting TIRGraph to Python code.
Framework-agnostic - works for all frontends.
"""
from ..core.graph import TIRGraph


def generate_forge_module(graph: TIRGraph, class_name="GeneratedForgeModule") -> str:
    """
    Generates a Python string for the Forge module by traversing the graph.
    """
    lines = []
    lines.append("import torch")
    lines.append("import forge")
    lines.append("")
    lines.append(f"class {class_name}(forge.Module):")
    lines.append(f"    def __init__(self, name='{graph.name}'):")
    lines.append(f"        super().__init__(name=name)")
    
    # 1. Add Parameters (Trainable weights)
    for name, tensor in graph.params.items():
        shape_str = str(tuple(tensor.shape))
        lines.append(f"        self.add_parameter('{name}', forge.Parameter(shape={shape_str}))")
    
    # 2. Add Constants (Non-trainable values)
    # Note: Constants are typically embedded directly in the graph, but we can also
    # add them as non-trainable parameters if needed
    for name, tensor in graph.constants.items():
        shape_str = str(tuple(tensor.shape))
        # Constants can be added as parameters with requires_grad=False, or embedded directly
        # For now, we'll add them as parameters for consistency
        lines.append(f"        self.add_parameter('{name}', forge.Parameter(shape={shape_str}, requires_grad=False))")
    lines.append("")
    
    # 3. Forward Method
    # Exclude both params and constants from forward arguments
    all_initializers = set(graph.params.keys()) | set(graph.constants.keys())
    forward_args = [inp for inp in graph.inputs if inp not in all_initializers]
    args_str = ", ".join(forward_args)
    lines.append(f"    def forward(self, {args_str}):")

    # 3. Operations
    sorted_nodes = graph.get_topological_sort()
    for node in sorted_nodes:
        op_info = node.emit()
        
        # Format Inputs
        inputs_str = ", ".join(op_info['input_names'])
        
        # Format Attributes (args from emit())
        attrs = op_info.get('args', {})
        attr_strs = []
        for k, v in attrs.items():
            if isinstance(v, str):
                attr_strs.append(f"{k}='{v}'")
            else:
                attr_strs.append(f"{k}={v}")
        attrs_str = ", ".join(attr_strs)
        
        call_args = inputs_str
        if attrs_str:
            call_args += f", {attrs_str}"
            
        # Format Output (single output per node - Forge constraint)
        output_name = op_info['output_name']
            
        lines.append(f"        # {node.op_type} -> {op_info['node_name']}")
        lines.append(f"        {output_name} = {op_info['function_name']}({call_args})")

    # 4. Return
    if len(graph.outputs) == 1:
        lines.append(f"        return {graph.outputs[0]}")
    else:
        out_str = ", ".join(graph.outputs)
        lines.append(f"        return {out_str}")
        
    return "\n".join(lines)

