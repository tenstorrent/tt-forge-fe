# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import json
from enum import Enum
from loguru import logger
from typing import Any, Dict, List, Optional

import torch

from forge.python_codegen import ForgeWriter
from forge.utils import create_excel_file


class NodeType(Enum):
    Activation = 1
    Parameter = 2
    Constant = 3

    @classmethod
    def to_json(cls, value):
        return value.name

    @classmethod
    def from_json(cls, value):
        return cls[value]


class Operation:
    """
    A class to store relevant code generation details about a specific operation.

    Attributes:
        function_name (str): The name of the function associated with the operation.
        node_name (str): The name of the node in the computation graph.
        output_name (str): The name of the output variable.
        input_names (list): A list of input variable names.
        input_shapes (list): A list of shapes corresponding to the input variables.
        input_dtypes (list): A list of dtypes corresponding to the input variables.
        args (list): A list of arguments for the operation.
        is_submodule_call (bool): A flag indicating if the operation is a submodule call (related to Torch 2.0).
        inputs_to_delete (list): A list of inputs to delete.
        loop_with (list): A list of loop variables.
        src_layer (optional): The source layer associated with the operation.
        metadata (dict): It contains additional information associated with the operation like model, variant, framework
    """

    def __init__(
        self,
        function_name,
        output_name="",
        node_name="",
        input_names=[],
        args=[],
        src_layer=None,
        input_shapes=[],
        input_dtypes=[],
        input_node_types=[],
        metadata={},
    ):
        self.function_name = function_name
        self.node_name = node_name
        self.output_name = output_name
        self.input_names = input_names
        self.input_node_types = input_node_types
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.args = args
        self.is_submodule_call = False
        self.inputs_to_delete = []
        self.loop_with = []
        self.src_layer = src_layer
        self.metadata = metadata


class OpArgs(dict):
    """
    OpArgs is dictionary subclass to store arguments in which argument name will be stored as dictionary key
    and argument values will be stored as dictionary values with additional utility methods for adding, removing,
    comparing and updating arguments.

    Methods:
        get_args_names: Returns a list of argument names.
        get_args_values: Returns a list of argument values.
        add_arg: Adds a new argument with a specified name and value.
        remove_arg: Removes an argument by name.
        update_arg: Updates the value of an existing argument by name.
        is_empty: Checks if the argument dictionary is empty.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_args_names(self):
        return list(self.keys())

    def get_args_values(self):
        return list(self.values())

    def add_arg(self, arg_name, arg_value):
        self[arg_name] = arg_value

    def remove_arg(self, arg_name):
        if arg_name in self:
            del self[arg_name]
        else:
            print(f"Arg '{arg_name}' not found.")

    def update_arg(self, arg_name, arg_value):
        if arg_name in self:
            self[arg_name] = arg_value
        else:
            print(f"Arg '{arg_name}' does not exist and cannot be updated.")

    def __eq__(self, other):
        if not isinstance(other, (OpArgs, dict)):
            return False

        for arg_name, arg_value in self.items():
            if arg_name in other.keys():
                if arg_value != other[arg_name]:
                    return False
            else:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_empty(self):
        return len(self) == 0

    def __str__(self):
        return super().__str__()


class OperandsInfo:
    """
    Stores operands information which include operand types(i.e NodeType), shapes and dtypes
    Args:
        operand_types: List of Operand type(i.e NodeType)
        operand_shapes: List of Operand shape
                        For constant nodetype, instead of storing shape, will store constant tensor.
        operand_dtypes: List of Operand datatype

    Methods:
        get_operand_types: Returns the list of operand types.
        get_operand_shapes: Returns the list of operand shapes.
        get_operand_dtypes: Returns the list of operand data types.
    """

    def __init__(self, operand_types, operand_shapes, operand_dtypes):

        # If lengths of operand_types, operand_shapes, and operand_dtypes doesn't match, raises assertion error
        assert len(operand_types) == len(operand_shapes) and len(operand_shapes) == len(
            operand_dtypes
        ), "Operands Type, shape, datatypes are not equal"
        self.operand_types = operand_types
        self.operand_shapes = operand_shapes
        self.operand_dtypes = operand_dtypes

    def get_operand_types(self):
        return self.operand_types

    def get_operand_shapes(self):
        return self.operand_shapes

    def get_operand_dtypes(self):
        return self.operand_dtypes

    def __eq__(self, other):
        """
        Checks equality between two OperandsInfo objects by comparing types, shapes, and data types.

        Args:
            other (OperandsInfo): The other OperandsInfo object to compare with.

        Returns:
            bool: True if both objects have the same operand information, otherwise False.
        """
        if not isinstance(other, OperandsInfo):
            return False
        if (
            len(self.operand_types) != len(other.operand_types)
            or len(self.operand_shapes) != len(other.operand_shapes)
            or len(self.operand_dtypes) != len(other.operand_dtypes)
        ):
            return False
        for type1, type2 in zip(self.operand_types, other.operand_types):
            if type1 != type2:
                return False
        for shape1, shape2 in zip(self.operand_shapes, other.operand_shapes):
            # For constant nodetype, will get constant tensor, instead of shape.
            if isinstance(shape1, torch.Tensor) and isinstance(shape2, torch.Tensor):
                if not torch.equal(shape1, shape2):
                    return False
            else:
                if shape1 != shape2:
                    return False
        for dtype1, dtype2 in zip(self.operand_dtypes, other.operand_dtypes):
            if dtype1 != dtype2:
                return False
        return True

    def __str__(self):
        if len(self.operand_types) > 0 and len(self.operand_shapes) > 0 and len(self.operand_dtypes) > 0:
            operands_info = "["
            for operand_type, operand_shape, operand_dtype in zip(
                self.operand_types, self.operand_shapes, self.operand_dtypes
            ):
                if isinstance(operand_shape, torch.Tensor):
                    operands_info += f"Operand(type={operand_type}, shape=Tensor, dtype={operand_dtype}), "
                else:
                    operands_info += f"Operand(type={operand_type}, shape={operand_shape}, dtype={operand_dtype}), "
            operands_info += "]"
            return operands_info

        else:
            return "OperandsInfo is empty!"


class OpArgsOpMetadata:
    """
    Stores Operation Args and associated metadata.

    Initializes OpArgsOpMetadata with a given OpArgs and operation metadata like operand_names.

    Args:
        args (OpArgs): The OpArgs object to associate with operation metadata.
        operation_metadata (Dict): Operation metadata to associate with args.

    Data Members:
        op_args_and_metadata (list of tuples): Each tuple contains an OpArgs object and a dict of operation metadata.
    """

    def __init__(self, args: OpArgs, operation_metadata: Dict[str, Any]):
        operation_metadata = self.transform_operation_metadata(operation_metadata)
        self.op_args_and_metadata = [(args, operation_metadata)]

    def get_op_args_and_metadata(self):
        return self.op_args_and_metadata

    def transform_operation_metadata(self, operation_metadata):
        new_operation_metadata = {}
        for name, value in operation_metadata.items():
            new_operation_metadata[name] = [value]
        return new_operation_metadata

    def update(self, new_args, new_operation_metadata):
        """
        Append Operation metadata if arguments match, otherwise adds new OpArgs and Operation metadata.

        Args:
            new_args (OpArgs): New arguments to match against existing ones.
            new_operation_metadata (list): New operation metadata to associate if new_args matches.
        """
        args_matched = False
        for idx, (arg, metadata) in enumerate(self.op_args_and_metadata):
            if (arg.is_empty() and new_args.is_empty()) or arg == new_args:
                for name, value in new_operation_metadata.items():
                    if value not in self.op_args_and_metadata[idx][1][name]:
                        self.op_args_and_metadata[idx][1][name].append(value)
                args_matched = True
                break
        if not args_matched:
            new_operation_metadata = self.transform_operation_metadata(new_operation_metadata)
            self.op_args_and_metadata.append((new_args, new_operation_metadata))

    def __str__(self):
        if len(self.op_args_and_metadata) > 0:
            op_args_and_metadata_info = ""
            for idx, (args, metadata) in enumerate(self.op_args_and_metadata, start=1):
                op_args_and_metadata_info += f"\t\t\t\t {idx})Opargs(" + str(args) + ")\n"
                for metadata_name, metadata_values in metadata.items():
                    op_args_and_metadata_info += f"\t\t\t\t\t\t" + str(metadata_name) + ":\n"
                    for metadata_value_idx, metadata_value in enumerate(metadata_values):
                        op_args_and_metadata_info += (
                            f"\t\t\t\t\t\t\t\t {metadata_value_idx})" + str(metadata_value) + "\n"
                        )
            return op_args_and_metadata_info
        else:
            return "OpArgsOpMetadata is empty!"


class UniqueOperationInfo:
    """
    Stores operands and argument associated with operation metadata.

    Args:
        operands (OperandsInfo): Information about operand types, shapes, and dtypes.
        opargs_opmetadata (OpArgsOpMetadata): Argument associated with the operation metadata.

    Data Members:
        unique_operands_and_opargs_opmetadata (list of tuples): Each tuple contains an OperandsInfo object
                                                             and an OpArgsOpMetadata object.
    """

    def __init__(self, operands: OperandsInfo, opargs_opmetadata: OpArgsOpMetadata):
        self.unique_operands_and_opargs_opmetadata = [(operands, opargs_opmetadata)]

    def get_unique_operands_and_opargs_opmetadata(self):
        return self.unique_operands_and_opargs_opmetadata

    def add_operands_args(self, new_operands, new_args, new_operation_metadata):
        """
        Adds or updates operandsInfo and Opargs and Operation metadata.

        Args:
            new_operands (OperandsInfo): Operands information.
            new_args (OpArgs): Operation arguments.
            new_operation_metadata (Dict): Operation metadata.
        """
        operands_matched = False
        for idx, (operands, opargs_opmetadata) in enumerate(self.unique_operands_and_opargs_opmetadata):
            if operands == new_operands:
                operands_matched = True
                self.unique_operands_and_opargs_opmetadata[idx][1].update(new_args, new_operation_metadata)
                break
        if not operands_matched:
            self.unique_operands_and_opargs_opmetadata.append(
                (new_operands, OpArgsOpMetadata(new_args, new_operation_metadata))
            )

    def __str__(self):
        if len(self.unique_operands_and_opargs_opmetadata) > 0:
            unique_operation_info = ""
            for idx, (operands, opargs_opmetadata) in enumerate(self.unique_operands_and_opargs_opmetadata, start=1):
                unique_operation_info += f"\t\t {idx})" + str(operands) + "\n"
                unique_operation_info += str(opargs_opmetadata) + "\n"
            return unique_operation_info

        else:
            return "UniqueOperationInfo is empty!"


class UniqueOperations(dict):
    """
    UniqueOperations is dictionary subclass to store forge op function name as dictionary key and
    UniqueOperationInfo (i.e Unique operands and Op arguments associated with operand names) as dictionary values
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def validate_node_types(cls, operand_names, operand_types, node_name_to_node_type):
        """
        Validates that each operand type matches the corresponding node type.

        Args:
            operand_names (list): Names of operands.
            operand_types (list): Types of operands.
            node_name_to_node_type (dict): Mapping of operand names to node types.

        Returns:
            bool: True if validation passes, otherwise False.
        """
        for operand_name, operand_type in zip(operand_names, operand_types):
            if operand_type == NodeType.Parameter and node_name_to_node_type[operand_name] != NodeType.Parameter:
                return False
            if operand_type == NodeType.Constant and node_name_to_node_type[operand_name] != NodeType.Constant:
                return False
            if operand_type == NodeType.Activation and node_name_to_node_type[operand_name] != NodeType.Activation:
                return False
        return True

    @classmethod
    def create_unique_operations(
        cls,
        ops: Dict[int, Operation],
        named_parameters: Dict[str, torch.Tensor],
        node_name_to_node_type: Optional[Dict[str, NodeType]] = None,
    ):
        """
        Creates unique operations by mapping operand and argument information to forge op names.

        Args:
            ops (dict): Dictionary of operation.
            named_parameters (dict): Mapping of node name to model parameters and buffers.
            node_name_to_node_type (dict): Mapping of node names to types.

        Returns:
            UniqueOperations: Populated UniqueOperations dictionary.
        """
        unique_operations = UniqueOperations()
        for nid in sorted(ops):
            forge_op_function_name = ops[nid].function_name
            operand_names = ops[nid].input_names
            operand_types = ops[nid].input_node_types
            if node_name_to_node_type is not None:
                assert UniqueOperations.validate_node_types(
                    operand_names, operand_types, node_name_to_node_type
                ), "Operand node types is not matching with node_name_to_node_type"
            operand_shapes = ops[nid].input_shapes
            operand_dtypes = ops[nid].input_dtypes
            args = ops[nid].args
            metadata = ops[nid].metadata
            operation_metadata = {"operand_names": operand_names}
            if len(metadata) != 0:
                for name, value in metadata.items():
                    operation_metadata[name] = value
            assert (
                len(operand_types) == len(operand_names)
                and len(operand_names) == len(operand_shapes)
                and len(operand_shapes) == len(operand_dtypes)
            ), "Operands names, shape, dtypes are not equal"

            # Replace constant node operand shape with constant value for comparing with other constant value.
            operand_shapes = [
                named_parameters[operand_name] if operand_type == NodeType.Constant else operand_shape
                for operand_type, operand_shape, operand_name in zip(operand_types, operand_shapes, operand_names)
            ]
            new_operands = OperandsInfo(operand_types, operand_shapes, operand_dtypes)
            new_args = OpArgs(args)
            if forge_op_function_name in unique_operations.keys():
                unique_operations[forge_op_function_name].add_operands_args(new_operands, new_args, operation_metadata)
            else:
                unique_operations[forge_op_function_name] = UniqueOperationInfo(
                    new_operands, OpArgsOpMetadata(new_args, operation_metadata)
                )

        return unique_operations

    def __str__(self):
        if len(self) > 0:
            unique_operations_info = ""
            for forge_op_function_name, unique_operation in self.items():
                unique_operations_info += forge_op_function_name + ": \n"
                unique_operations_info += str(unique_operation) + "\n"
            return unique_operations_info
        else:
            return "UniqueOperations is empty!"


def export_unique_op_configuration_info(module_name, unique_operation_data, unique_ops_metadata):
    headers = ["Op", "Operand_Names", "Operand_Shapes", "Operand_Types", "Operand_Dtypes", "Args", "Testfile"]
    rows = []
    for operation_info in unique_operation_data:
        rows.append([operation_info[header] for header in headers])

    export_tvm_generated_unique_op_tests_details_dir_path = os.getenv(
        "FORGE_EXPORT_TVM_GENERATED_UNIQUE_OP_TESTS_DETAILS_DIR_PATH", f"generated_modules/unique_ops/"
    )
    export_tvm_generated_unique_op_tests_details_dir_path = os.path.join(
        export_tvm_generated_unique_op_tests_details_dir_path, module_name
    )
    if not os.path.exists(export_tvm_generated_unique_op_tests_details_dir_path):
        os.makedirs(export_tvm_generated_unique_op_tests_details_dir_path)

    export_tvm_generated_unique_op_tests_details_file_path = os.path.join(
        export_tvm_generated_unique_op_tests_details_dir_path,
        "tvm_generated_unique_op_test_details.xlsx",
    )

    unique_ops_metadata_path = os.path.join(
        export_tvm_generated_unique_op_tests_details_dir_path,
        "tvm_generated_unique_ops_metadata.json",
    )
    with open(unique_ops_metadata_path, "w") as json_file:
        json.dump(unique_ops_metadata, json_file, indent=4)

    create_excel_file(
        title=module_name,
        headers=headers,
        rows=rows,
        output_file_path=export_tvm_generated_unique_op_tests_details_file_path,
    )


def generate_unique_op_tests(
    ops,
    current_module_name,
    framework,
    contains_incompatible_np_floats,
    delete_inputs,
    node_name_to_node_type,
    params,
    constants,
    param_names,
    param_file_name,
    named_params_file_name,
    named_buffers_file_name,
    compiler_cfg,
):
    """
    Generates test modules for unique operation configurations.

    The function extracts unique operation configurations based on the operation names, operand types, shapes,
    and datatypes, as well as operation arguments (if any). For operation, a test module
    file is created, which includes a Forge module for different configurations and associated test cases.
    """

    # Load the named parameters, constants, and buffers from files
    named_parameters = torch.load(named_params_file_name)
    if param_file_name is not None:
        serialized_params = torch.load(param_file_name)
        named_parameters.update(serialized_params)
    named_buffers = torch.load(named_buffers_file_name)
    named_parameters.update(named_buffers)

    # Extract unique operations by comparing operands types, shapes and dtypes and arguments if any
    unique_operations = UniqueOperations.create_unique_operations(ops, named_parameters, node_name_to_node_type)

    def get_param_const(name):
        for nid, param in params.items():
            if param[0] == name:
                return nid, param
        for nid, const in constants.items():
            if const[0] == name:
                return nid, const
        logger.error(f"There is no paramter/constant with the name {name}")

    unique_operation_details = []
    for op_idx, forge_op_function_name in enumerate(sorted(unique_operations)):

        # Extract operation name from forge op function name
        op_name = forge_op_function_name.split(".")[-1].lower()

        module_name = "test_" + op_name

        # Initialize Forge writer and generate header with pytest specific imports
        writer = ForgeWriter(
            module_name,
            framework,
            module_directory=f"generated_modules/unique_ops/{current_module_name}",
            contains_incompatible_np_floats=contains_incompatible_np_floats,
            delete_inputs=delete_inputs,
        )
        writer.write_header(include_pytest_imports=True)

        # Get the unique operands and operation arguments assiocated the operand names
        unique_operands_and_opargs_opmetadata = unique_operations[
            forge_op_function_name
        ].get_unique_operands_and_opargs_opmetadata()

        pytest_input_shapes_and_dtypes_list = []
        forge_module_names = []
        module_idx = 0
        forge_module_list = []
        test_count = 0
        for operands_idx, (operands, opargs_opmetadata) in enumerate(unique_operands_and_opargs_opmetadata):

            for args_idx, (args, operation_metadata) in enumerate(opargs_opmetadata.get_op_args_and_metadata()):

                operand_types = operands.get_operand_types()
                operand_shapes = operands.get_operand_shapes()
                operand_dtypes = operands.get_operand_dtypes()
                operand_names = operation_metadata["operand_names"][0]

                if compiler_cfg.export_tvm_generated_unique_op_tests_details:
                    operation_info = {}
                    operation_info["Op"] = forge_op_function_name
                    operation_info["Operand_Names"] = str(operand_names)
                    operation_info["Operand_Shapes"] = str(
                        [
                            operand_name if operand_type == NodeType.Constant else operand_shape
                            for operand_type, operand_shape, operand_name in zip(
                                operand_types, operand_shapes, operand_names
                            )
                        ]
                    )
                    operation_info["Operand_Types"] = str(
                        [NodeType.to_json(operand_type) for operand_type in operand_types]
                    )
                    operation_info["Operand_Dtypes"] = str(operand_dtypes)
                    operation_info["Args"] = str(args)

                # Check if all operands types are parameters or constants and change the operand type from
                # parameters or constants to activation and pass it as activation to forge module forward function
                all_params_const = all(
                    [
                        True if (operand_type == NodeType.Parameter or operand_type == NodeType.Constant) else False
                        for operand_type in operand_types
                    ]
                )
                if all_params_const:
                    operand_types = [NodeType.Activation] * len(operand_types)
                    operand_shapes = operand_names
                    operand_names = [op_name + "_input_" + str(idx) for idx in range(len(operand_names))]

                # Check if an existing Forge module matches the current operation configuration.
                # This involves comparing the number of inputs, operand types, activation operand count,
                # and arguments. If a match is found, further checks are made to ensure that the parameter
                # shapes and data types, or constants, match as well. If a match is found for either parameters
                # or constants, the new Forge module creation is skipped. If no match is found, a new Forge module
                # will be created for the current operation configuration.
                need_to_create_forge_module = True
                for forge_mod in forge_module_list:
                    if (
                        len(forge_mod["operand_types"]) == len(operand_types)
                        and forge_mod["operand_types"] == operand_types
                    ):
                        if (
                            forge_mod["number_of_activation"]
                            == len(
                                list(filter(lambda operand_type: operand_type == NodeType.Activation, operand_types))
                            )
                            and forge_mod["args"] == args
                        ):
                            param_shape_dtype_list = [
                                (operand_shape, operand_dtype)
                                for operand_type, operand_shape, operand_dtype in zip(
                                    operand_types, operand_shapes, operand_dtypes
                                )
                                if operand_type == NodeType.Parameter
                            ]

                            const_list = [
                                operand_shape
                                for operand_type, operand_shape in zip(operand_types, operand_shapes)
                                if operand_type == NodeType.Constant
                            ]

                            if forge_mod["number_of_parameters"] > 0 and len(param_shape_dtype_list) > 0:
                                if len(param_shape_dtype_list) == forge_mod["number_of_parameters"]:
                                    params_shape_dtype_equal = all(
                                        [
                                            True if (shape1 == shape2 and dtype1 == dtype2) else False
                                            for (shape1, dtype1), (shape2, dtype2) in zip(
                                                forge_mod["param_shape_dtype_list"], param_shape_dtype_list
                                            )
                                        ]
                                    )
                                    if params_shape_dtype_equal:
                                        need_to_create_forge_module = False
                                        forge_module_names.append(forge_mod["class_name"])
                                        break
                            elif forge_mod["number_of_constants"] > 0 and len(const_list) > 0:
                                if len(const_list) == forge_mod["number_of_constants"]:
                                    const_equal = all(
                                        [
                                            True if torch.equal(const1, const2) else False
                                            for const1, const2 in zip(forge_mod["const_list"], const_list)
                                        ]
                                    )
                                    if const_equal:
                                        need_to_create_forge_module = False
                                        forge_module_names.append(forge_mod["class_name"])
                                        break
                            else:
                                need_to_create_forge_module = False
                                forge_module_names.append(forge_mod["class_name"])
                                break

                # If no matching Forge module was found, create a new one for the current operation configuration
                if need_to_create_forge_module:

                    # Generate class name and append it forge_module_names list for using it as pytest parameter.
                    class_name = current_module_name.lower() + op_name + str(module_idx)
                    class_name = class_name.title().replace("_", "")
                    forge_module_names.append(class_name)

                    needed_params = {}
                    needed_consts = {}
                    params_shape_dtype_list = []
                    const_list = []
                    forward_method_inputs = {}
                    new_operand_names = []

                    # Iterate through operand types and names to classify them as parameters, constants, or activations.
                    # Collect the necessary parameters and constants, and use them to generate the class definition and
                    # handle activations for the forward method inputs.
                    for idx, (operand_type, operand_name) in enumerate(zip(operand_types, operand_names)):
                        if operand_type == NodeType.Parameter:
                            nid, param_tuple = get_param_const(operand_name)
                            needed_params[nid] = param_tuple
                            params_shape_dtype_list.append([param_tuple[1], param_tuple[3]])
                            new_operand_names.append(operand_name)
                        elif operand_type == NodeType.Constant:
                            nid, const_tuple = get_param_const(operand_name)
                            needed_consts[nid] = const_tuple
                            const_list.append(named_parameters[operand_name])
                            new_operand_names.append(operand_name)
                        else:
                            if operand_name not in forward_method_inputs.values():
                                forward_method_inputs[idx] = operand_name
                            else:
                                forward_method_inputs[idx] = op_name + "_input_" + str(idx)
                                logger.warning(
                                    f"operand_name {operand_name} is already present in the forward_method_inputs {forward_method_inputs}"
                                )
                            new_operand_names.append(forward_method_inputs[idx])

                    # Generate the class definition with the collected parameters and constants.
                    writer.write_class_definition(params=needed_params, constants=needed_consts, class_name=class_name)

                    # Create a single operation with the function name, output name,
                    # input operand names, and arguments and use it for generating forward method
                    single_op = {
                        args_idx: Operation(
                            function_name=forge_op_function_name,
                            output_name=op_name + "_output_1",
                            input_names=new_operand_names,
                            args=tuple(args.items()),
                        )
                    }

                    forward_method_returns = {args_idx: single_op[args_idx].output_name}

                    # Generate forge module forward function
                    writer.write_forward(single_op, forward_method_inputs, forward_method_returns)

                    # If there are any parameters or constants, generate the parameter parser function.
                    if len(needed_params) != 0 or len(needed_consts) != 0:
                        writer.write_param_parser(
                            param_names, param_file_name, named_params_file_name, named_buffers_file_name
                        )

                    module_idx += 1
                    forge_module_list.append(
                        {
                            "class_name": class_name,
                            "operand_types": operand_types,
                            "number_of_activation": len(forward_method_inputs),
                            "number_of_parameters": len(needed_params),
                            "number_of_constants": len(needed_consts),
                            "param_shape_dtype_list": params_shape_dtype_list,
                            "const_list": const_list,
                            "args": args,
                        }
                    )

                # Collect activation input shapes and dtypes for using it in pytest parameter
                pytest_input_shapes_dtypes = []
                for operand_type, operand_shape, operand_dtype in zip(operand_types, operand_shapes, operand_dtypes):
                    if operand_type == NodeType.Activation:
                        pytest_input_shapes_dtypes.append((operand_shape, operand_dtype))
                pytest_input_shapes_and_dtypes_list.append(pytest_input_shapes_dtypes)

                if compiler_cfg.export_tvm_generated_unique_op_tests_details:
                    operation_info["Testfile"] = (
                        writer.module_directory
                        + "/"
                        + writer.filename
                        + f"::test_module[forge_module_and_shapes_dtypes{test_count}]"
                    )
                    unique_operation_details.append(operation_info)
                    test_count += 1

        # If the parameter/constant is passed as activation, operand shape will be replaced with operand name
        # because instead of generating tensor from shape, use actual tensor from model parameters/buffers
        # and so generating function for loading the model parameters/buffers and saving it as named_parameter variable
        need_model_parameter_function = any(
            [
                True if isinstance(shape, str) else False
                for pytest_input_shapes_dtypes in pytest_input_shapes_and_dtypes_list
                for shape, _ in pytest_input_shapes_dtypes
            ]
        )
        if need_model_parameter_function:
            writer.write_model_parameter_function(param_file_name, named_params_file_name, named_buffers_file_name)

        # Generate pytest function for the operation with pytest parameter containing list of tuple
        # and each tuple constaints module name, tuple of operand shape/name and dtype
        writer.write_pytest_function(
            forge_module_names,
            framework,
            pytest_input_shapes_and_dtypes_list,
        )

        writer.close_file()

        if compiler_cfg.export_tvm_generated_unique_op_tests_details:
            unique_ops_metadata = {
                "framework": framework,
                "module_name": current_module_name,
                "param_file_name": param_file_name,
                "named_params_file_name": named_params_file_name,
                "named_buffers_file_name": named_buffers_file_name,
            }
            export_unique_op_configuration_info(current_module_name, unique_operation_details, unique_ops_metadata)
