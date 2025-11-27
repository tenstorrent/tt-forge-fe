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
    
    # 1. Add Parameters (Initializers)
    for name, tensor in graph.initializers.items():
        shape_str = str(tuple(tensor.shape))
        lines.append(f"        self.add_parameter('{name}', forge.Parameter(shape={shape_str}))")
    lines.append("")

    # 2. Forward Method
    forward_args = [inp for inp in graph.inputs if inp not in graph.initializers]
    args_str = ", ".join(forward_args)
    lines.append(f"    def forward(self, {args_str}):")

    # 3. Operations
    sorted_nodes = graph.get_topological_sort()
    for node in sorted_nodes:
        op_info = node.emit()
        
        # Format Inputs
        inputs_str = ", ".join(op_info['inputs'])
        
        # Format Attributes
        attrs = op_info.get('attrs', {})
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
            
        # Format Outputs
        outputs = op_info['outputs']
        if len(outputs) == 1:
            lhs = outputs[0]
        else:
            lhs = ", ".join(outputs)
            
        lines.append(f"        # {node.op_type} -> {op_info['op_name']}")
        lines.append(f"        {lhs} = {op_info['forge_func']}({call_args})")

    # 4. Return
    if len(graph.outputs) == 1:
        lines.append(f"        return {graph.outputs[0]}")
    else:
        out_str = ", ".join(graph.outputs)
        lines.append(f"        return {out_str}")
        
    return "\n".join(lines)

