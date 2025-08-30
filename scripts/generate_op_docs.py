# scripts/docgen/generate_op_docs.py
"""
Generates a Markdown document listing supported operators.
"""

import os
from forge.op_repo.datatypes import OperatorDefinition
from forge.op_repo.forge_operators import forge_operator_repository
from forge.op_repo.pytorch_operators import pytorch_operator_repository


def format_parameters(op_def: OperatorDefinition, param_type: str) -> str:
    """
    Formats operator parameters into a readable string.

    Args:
        op_def: The OperatorDefinition object.
        param_type: The type of parameters to format ("constructor" or "forward").

    Returns:
        A string representation of the formatted parameters.
    """
    params = []
    if param_type == "constructor":
        param_defs = op_def.constructor_params
    elif param_type == "forward":
        param_defs = op_def.forward_params
    else:
        return "Invalid parameter type"

    for p_def in param_defs:
        # Adjust the following lines based on the actual structure of OperatorParamNumber
        param_str = f"{p_def.name} ({getattr(p_def, 'type', 'unknown')})"
        if hasattr(p_def, "min_value") and hasattr(p_def, "max_value") and p_def.min_value is not None and p_def.max_value is not None:
            param_str += f": [{p_def.min_value}, {p_def.max_value}]"
        params.append(param_str)
    return ", ".join(params) if params else "N/A"


# Define the output directory and filename
OUTPUT_DIR = "docs/src/ops/"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "supported_ops.md")


def create_output_directory():
    """Creates the output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_markdown_table(title: str, ops: list[OperatorDefinition]) -> str:
    """
    Generates a Markdown table for a list of operators.

    Args:
        title: The title of the table section.
        ops: A list of OperatorDefinition objects.

    Returns:
        A string containing the Markdown table.
    """
    markdown = f"## {title}\n\n"
    markdown += "| Name | Full Name / Path | Inputs | Constructor Parameters | Forward Parameters |\n"
    markdown += "|------|--------------------|--------|------------------------|--------------------|\n"

    for op_def in ops:
        inputs = f"{op_def.input_num_range.operands_min}-{op_def.input_num_range.operands_min}"
        constructor_params = format_parameters(op_def, "constructor")
        forward_params = format_parameters(op_def, "forward")
        markdown += f"| {op_def.name} | {op_def.full_name} | {inputs} | {constructor_params} | {forward_params} |\n"
    
    markdown += "\n"
    return markdown


def generate_markdown_content() -> str:
    """
    Generates the full Markdown content for the supported operators document.

    Returns:
        A string containing the complete Markdown document.
    """
    content = "# Supported Operators\n\n"
    
    # Forge Operators
    content += generate_markdown_table("Forge Operators", forge_operator_repository.operators)
    
    # PyTorch Operators
    content += generate_markdown_table("PyTorch Operators", pytorch_operator_repository.operators)
    
    return content


def write_markdown_to_file(content: str):
    """
    Writes the Markdown content to the output file.

    Args:
        content: The Markdown content to write.
    """
    try:
        with open(OUTPUT_FILENAME, "w") as f:
            f.write(content)
        print(f"Successfully generated {OUTPUT_FILENAME}")
    except IOError as e:
        print(f"Error writing to file {OUTPUT_FILENAME}: {e}")


if __name__ == "__main__":
    try:
        create_output_directory()
        markdown_content = generate_markdown_content()
        write_markdown_to_file(markdown_content)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
