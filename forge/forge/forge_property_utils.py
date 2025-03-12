# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from dataclasses import dataclass, is_dataclass, field
from dataclasses_json import dataclass_json
from typing import Union, List, Optional, Any, get_origin, get_args, Dict
from forge.verify.config import VerifyConfig
from forge.config import CompilerConfig
from forge._C import ExecutionDepth


class ExecutionStage(Enum):
    FAILED_TVM_RELAY_IRMODULE_GENERATION = 0
    FAILED_TVM_RELAY_IO_FLATTENING = 1
    FAILED_TVM_RELAY_IR_TRANSFORMATION = 2
    FAILED_TVM_PATTERN_CALLBACKS = 3
    FAILED_TVM_GRAPH_PARTITIONING = 3
    FAILED_FORGE_MODULE_GENERATION = 4
    FAILED_FORGE_INITIAL_GRAPH_PASS = 5
    FAILED_FORGE_POST_INITIAL_GRAPH_PASS = 6
    FAILED_FORGE_CONSTEVAL = 7
    FAILED_FORGE_OPTIMIZATION_GRAPH_PASS = 8
    FAILED_FORGE_POST_OPTIMIZATION_DECOMP = 9
    FAILED_FORGE_AUTOGRAD_PASS = 10
    FAILED_FORGE_POST_AUTOGRAD_DECOMP = 11
    FAILED_FORGE_PRE_LOWERING = 12
    FAILED_FORGE_GRAPH_SPLIT = 13
    FAILED_FORGE_MLIR_COMPILATION = 14
    FAILED_TTNN_BINARY_EXECUTION = 15
    FAILED_VERIFICATION = 16
    PASSED = 17

    @classmethod
    def to_json(cls, value):
        return value.name

    @classmethod
    def from_json(cls, value):
        return cls[value.upper()]


@dataclass_json
@dataclass
class Config:
    compiler: Dict[str, Any] = field(default_factory=lambda: dict())
    verify: Dict[str, Any] = field(default_factory=lambda: dict())


@dataclass_json
@dataclass
class Tags:
    model_name: Optional[Union[List[str], str]] = None
    op_name: str = ""
    bringup_status: str = ""
    pcc: Optional[List[float]] = None
    atol: Optional[List[float]] = None
    execution_stage: str = ""
    config: Optional[Config] = None


@dataclass_json
@dataclass
class ForgePropertyStore:
    owner: str = "tt-forge-fe"
    group: str = ""
    tags: Optional[Tags] = None


class ForgePropertyHandler:
    """
    Handles storage and retrieval of properties in a nested ForgePropertyStore.

    This class provides methods to add, update, retrieve, and clean properties stored in a
    ForgePropertyStore. It supports nested attributes using dot-separated keys and includes
    several utility methods for recording specific property values (such as group, model name,
    and configuration data).

    Attributes:
        store (ForgePropertyStore): The underlying store containing the property data.
    """

    def __init__(self, store: ForgePropertyStore):
        self.store = store

    def add(self, key: str, value: Any):
        """
        Adds or updates a property in the store using a dot-separated key.

        The dot-separated key indicates the nested structure. For example, a key "tags.model_name"
        means that the 'model_name' attribute is under the 'tags' attribute of the store. If any
        intermediate attribute is missing or is None, it is created using a default instance.

        Args:
            key (str): Dot-separated string representing the property path.
            value (Any): The value to be assigned at the specified property path.

        Raises:
            ValueError: If the provided key is empty.
        """
        if not key:
            raise ValueError("Key cannot be empty.")
        keys = key.split(".")
        obj = self.store
        # Traverse the nested structure, creating missing attributes as needed.
        for k in keys[:-1]:
            if not hasattr(obj, k) or getattr(obj, k) is None:
                default_instance = self._create_default_instance(obj, k)
                setattr(obj, k, default_instance)
            obj = getattr(obj, k)
        # Set the final attribute to the provided value.
        setattr(obj, keys[-1], value)

    def __call__(self, key: str, value: Any):
        """
        Allows the handler to be called like a function to add or update a property.

        Example:
            handler("tags.op_name", "relu")

        Args:
            key (str): Dot-separated key path for the property.
            value (Any): The value to be set.
        """
        self.add(key, value)

    def get(self, key: str):
        """
        Retrieves a property value using a dot-separated key.

        The function traverses the nested attributes as specified by the key and returns the value.
        If any attribute in the chain is missing or is None, the function returns None.

        Args:
            key (str): Dot-separated string representing the property path.

        Returns:
            Any: The value of the property, or None if any attribute in the path is missing.

        Raises:
            ValueError: If the provided key is empty.
        """
        if not key:
            raise ValueError("Key cannot be empty.")
        keys = key.split(".")
        obj = self.store
        # Walk through the nested structure.
        for k in keys:
            if not hasattr(obj, k):
                return None
            obj = getattr(obj, k)
            if obj is None:
                return None
        return obj

    def _create_default_instance(self, parent: Any, attr: str):
        """
        Creates a default instance for a missing attribute based on the parent's dataclass metadata.

        For Optional fields that have a dataclass as their inner type, this method will instantiate
        the inner type. If a suitable dataclass cannot be inferred, it falls back to returning an empty dict.

        Args:
            parent (Any): The parent object, typically a dataclass instance.
            attr (str): The name of the attribute for which a default instance is needed.

        Returns:
            Any: A new instance of the attribute's type, or an empty dictionary if instantiation fails.
        """
        if hasattr(parent, "__dataclass_fields__"):
            field_info = parent.__dataclass_fields__.get(attr)
            if field_info is not None:
                field_type = field_info.type
                # Handle Optional[T] by extracting T from Union[T, None].
                origin = get_origin(field_type)
                if origin is Union:
                    args = get_args(field_type)
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if len(non_none_types) == 1:
                        field_type = non_none_types[0]
                if is_dataclass(field_type):
                    return field_type()
        # Fallback: return an empty dictionary if no dataclass instance can be created.
        return {}

    def record_group(self, group: str):
        """
        Records the group property in the tags.

        Args:
            group (str): The group value to be recorded.
        """
        self.add("group", group)

    def record_model_name(self, model_name: Union[str, List[str]]):
        """
        Records the model name in the tags.

        Args:
            model_name (Union[str, List[str]]): The model name (or list of model names) to record.
        """
        self.add("tags.model_name", model_name)

    def record_op_name(self, op_name: str):
        """
        Records the operation name in the tags.

        Args:
            op_name (str): The operation name to be recorded.
        """
        self.add("tags.op_name", op_name)

    def record_pcc(self, pcc: List[float]):
        """
        Records the PCC metric in the tags.

        Args:
            pcc (List[float]): A list of PCC values.
        """
        self.add("tags.pcc", pcc)

    def record_atol(self, atol: List[float]):
        """
        Records the atol (absolute tolerance) values in the tags.

        Args:
            atol (List[float]): A list of atol values.
        """
        self.add("tags.atol", atol)

    def record_pcc_and_atol(self, pcc: List[float], atol: List[float]):
        """
        Records both PCC and atol values in the tags.

        Args:
            pcc (List[float]): A list of PCC values.
            atol (List[float]): A list of atol values.
        """
        self.record_pcc(pcc)
        self.record_atol(atol)

    def record_execution_depth(self, execution_depth: ExecutionDepth):
        """
        Records the execution depth (as bringup_status) in the tags.

        Args:
            execution_depth (ExecutionDepth): The execution depth value, converted to JSON.
        """
        self.add("tags.bringup_status", ExecutionDepth.to_json(execution_depth))

    def record_execution_stage(self, execution_stage: ExecutionStage):
        """
        Records the execution stage in the tags.

        Args:
            execution_stage (ExecutionStage): The execution stage value, converted to JSON.
        """
        self.add("tags.execution_stage", ExecutionStage.to_json(execution_stage))

    def record_compiler_config(self, compiler_config: CompilerConfig):
        """
        Records the compiler configuration under tags.config.compiler.

        Args:
            compiler_config (CompilerConfig): The compiler configuration object.
        """
        self.add("tags.config.compiler", compiler_config.to_dict())

    def record_verify_config(self, verify_config: VerifyConfig):
        """
        Records the verify configuration under tags.config.verify.

        Converts the verify configuration to a dictionary, and ensures that the value
        for the 'value_checker' key is also represented as a dictionary.

        Args:
            verify_config (VerifyConfig): The verify configuration object.
        """
        verify_config = verify_config.to_dict()
        verify_config["value_checker"] = verify_config["value_checker"].__dict__
        self.add("tags.config.verify", verify_config)

    def to_dict(self):
        """
        Converts the entire property store to a dictionary.

        Returns:
            dict: The property store represented as a dictionary.
        """
        return self.store.to_dict()

    def items(self):
        """
        Returns an iterator over the property store items.

        Returns:
            ItemsIterator: An iterator over the key-value pairs in the property store.
        """
        return self.to_dict().items()

    def clean_store(self) -> dict:
        """
        Returns a cleaned dictionary version of the underlying store.

        A value is considered empty if it is None or an empty container (empty string, list,
        tuple, dict, or set). All keys with empty values are removed, except for keys named
        "compiler" or "verify". For those keys, the entire dict value is preserved as is,
        even if it contains empty members.

        Returns:
            dict: A cleaned version of the property store with empty values removed.
        """

        def is_empty(value):
            # Returns True if the value is None or an empty container.
            if value is None:
                return True
            if isinstance(value, (str, list, tuple, dict, set)) and len(value) == 0:
                return True
            return False

        def recursive_clean(data):
            # Recursively cleans a dictionary by removing keys with empty values.
            if not isinstance(data, dict):
                return data
            cleaned = {}
            for key, value in data.items():
                # Preserve keys "compiler" and "verify" as is.
                if key in ("compiler", "verify"):
                    cleaned[key] = value
                    continue

                if isinstance(value, dict):
                    cleaned_value = recursive_clean(value)
                else:
                    cleaned_value = value

                if not is_empty(cleaned_value):
                    cleaned[key] = cleaned_value
            return cleaned

        store_dict = self.store.to_dict()
        return recursive_clean(store_dict)

    def store_property(self, record_property):
        """
        Stores the cleaned properties using a provided recording function.

        Args:
            record_property (Callable): A function that accepts two arguments (property_name and
                property_value) to record the value.
        """
        cleaned_property_store = self.clean_store()
        for property_name, property_value in cleaned_property_store.items():
            record_property(property_name, property_value)
