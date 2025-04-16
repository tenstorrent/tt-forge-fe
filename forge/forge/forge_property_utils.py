# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
import json
import re
from dataclasses import dataclass, is_dataclass, field
from dataclasses_json import dataclass_json
from typing import Union, List, Optional, Any, get_origin, get_args, Dict, Tuple
from forge.verify.config import VerifyConfig
from forge.config import CompilerConfig
from forge._C import ExecutionDepth, DataFormat
from forge.tensor import Tensor, forge_dataformat_to_pytorch_dtype
from loguru import logger
from forge.module import ForgeModule
import forge


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class BaseEnum(Enum):
    """Base Enum to handle short and full name attributes."""

    def __init__(self, short: str, full: str):
        self.short = short  # Short identifier (e.g., "hf" for Hugging Face)
        self.full = full  # Full descriptive name (e.g., "Hugging Face")


class Framework(BaseEnum):
    PYTORCH = ("pt", "PyTorch")
    TENSORFLOW = ("tf", "TensorFlow")
    ONNX = ("onnx", "ONNX")
    PADDLE = ("pd", "PaddlePaddle")
    JAX = ("jax", "JAX")


class Task(BaseEnum):
    SPEECH_TRANSLATE = ("speech_translate", "Speech Translation")
    MUSIC_GENERATION = ("music_generation", "Music Generation")
    SPEECH_RECOGNITION = ("speech_recognition", "Speech Recognition")
    QA = ("qa", "Question Answering")
    MASKED_LM = ("mlm", "Masked Language Modeling")
    CAUSAL_LM = ("clm", "Causal Language Modeling")
    TOKEN_CLASSIFICATION = ("token_cls", "Token Classification")
    SEQUENCE_CLASSIFICATION = ("seq_cls", "Sequence Classification")
    IMAGE_CLASSIFICATION = ("img_cls", "Image Classification")
    IMAGE_SEGMENTATION = ("img_seg", "Image Segmentation")
    POSE_ESTIMATION = ("pose_estimation", "Pose Estimation")
    DEPTH_PREDICTION = ("depth_prediction", "Depth Prediction")
    TEXT_GENERATION = ("text_gen", "Text Generation")
    OBJECT_DETECTION = ("obj_det", "Object Detection")
    SEMANTIC_SEGMENTATION = ("sem_seg", "Semantic Segmentation")
    MASKED_IMAGE_MODELING = ("masked_img", "Masked Image Modeling")
    CONDITIONAL_GENERATION = ("cond_gen", "Conditional Generation")
    IMAGE_ENCODING = ("img_enc", "Image Encoding")
    VISUAL_BACKBONE = ("visual_bb", "Visual Backbone")
    DEPTH_ESTIMATION = ("depth_estimation", "Depth Estimation")
    SCENE_TEXT_RECOGNITION = ("scene_text_recognition", "Scene Text Recognition")
    TEXT_TO_SPEECH = ("text_to_speech", "Text to Speech")
    SENTENCE_EMBEDDING_GENERATION = ("sentence_embed_gen", "Sentence Embedding Generation")
    MULTIMODAL_TEXT_GENERATION = ("multimodal_text_gen", "Multimodal Text Generation")
    ATOMIC_ML = ("atomic_ml", "Atomic Machine Learning")


class Source(BaseEnum):
    HUGGINGFACE = ("hf", "Hugging Face")
    TORCH_HUB = ("torchhub", "Torch Hub")
    TIMM = ("timm", "TIMM")
    OSMR = ("osmr", "OSMR")
    TORCHVISION = ("torchvision", "Torchvision")
    GITHUB = ("github", "GitHub")
    PADDLE = ("paddlemodels", "Paddle Models")


def build_module_name(
    framework: Framework,
    model: str,
    task: Task,
    source: Source,
    variant: str = "base",
    suffix: str | None = None,
) -> str:
    module_name = f"{framework}_{model}_{variant}_{task}_{source}"
    if suffix is not None:
        module_name += f"_{suffix}"
    module_name = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
    module_name = re.sub(r"_+", "_", module_name)
    module_name = module_name.lower()
    return module_name


class ExecutionStage(Enum):
    FAILED_BEFORE_FORGE_COMPILATION_INITIATION = auto()
    FAILED_TVM_RELAY_IRMODULE_GENERATION = auto()
    FAILED_TVM_RELAY_IO_FLATTENING = auto()
    FAILED_TVM_RELAY_IR_TRANSFORMATION = auto()
    FAILED_TVM_PATTERN_CALLBACKS = auto()
    FAILED_TVM_GRAPH_PARTITIONING = auto()
    FAILED_FORGE_MODULE_GENERATION = auto()
    FAILED_FORGE_INITIAL_GRAPH_PASS = auto()
    FAILED_FORGE_POST_INITIAL_GRAPH_PASS = auto()
    FAILED_FORGE_CONSTEVAL = auto()
    FAILED_FORGE_OPTIMIZATION_GRAPH_PASS = auto()
    FAILED_FORGE_POST_OPTIMIZATION_DECOMP = auto()
    FAILED_FORGE_AUTOGRAD_PASS = auto()
    FAILED_FORGE_POST_AUTOGRAD_DECOMP = auto()
    FAILED_FORGE_PRE_LOWERING = auto()
    FAILED_FORGE_GRAPH_SPLIT = auto()
    FAILED_FORGE_MLIR_COMPILATION = auto()
    FAILED_TTNN_BINARY_EXECUTION = auto()
    FAILED_VERIFICATION = auto()
    PASSED = auto()

    @classmethod
    def to_str(cls, value):
        return value.name

    @classmethod
    def from_str(cls, value):
        return cls[value.upper()]


@dataclass_json
@dataclass
class TensorDesc:
    shape: List[int]
    data_type: str = ""
    buffer_type: str = ""
    layout: str = ""
    grid_shape: Optional[List[int]] = None


class FlatbufferDetailsExtractor:
    """
    A utility class to parse and extract comprehensive details from a generated flatbuffer binary JSON.

    Args:
        binary_json (Dict[str, Any]): The flatbuffer binary JSON containing program details.
    """

    def __init__(self, binary_json):
        self.binary = binary_json

    def extract_tensor_details(self, inputs_or_outputs):
        """
        Extracts tensor descriptions from a list of input/output entries.

        Parameters:
            inputs_or_outputs (list): A list of dictionaries, each containing a "desc" key
                                      with tensor description details.

        Returns:
            list: A list of TensorDesc objects.
        """
        tensor_desc_list = []
        for input_or_output in inputs_or_outputs:
            desc = input_or_output["desc"]
            if (
                "shape" in desc
                and "layout" in desc
                and "memory_desc" in desc["layout"]
                and "data_type" in desc["layout"]["memory_desc"]
            ):
                tensor_desc = TensorDesc(shape=desc["shape"], data_type=desc["layout"]["memory_desc"]["data_type"])

                try:
                    tensor_desc.buffer_type = desc["layout"]["memory_desc"]["memory_space"]
                except KeyError:
                    if "memory_config" in desc["layout"]["memory_desc"]:
                        # If the tensor is on device, the descriptor will have a "memory_config" field.
                        tensor_desc.buffer_type = desc["layout"]["memory_desc"]["memory_config"]["buffer_type"]
                    else:
                        # If the tensor is on host, the descriptor will have a "storage_type" field.
                        tensor_desc.buffer_type = desc["layout"]["memory_desc"]["storage_type"]

                try:
                    tensor_desc.layout = desc["layout"]["memory_desc"]["memory_layout"]
                except KeyError:
                    if "memory_config" in desc["layout"]["memory_desc"]:
                        # If the tensor is on device, use the tensor_memory_layout from memory_config.
                        tensor_desc.layout = desc["layout"]["memory_desc"]["memory_config"]["tensor_memory_layout"]
                    else:
                        # If the tensor is on host, no tensor_memory_layout is available.
                        tensor_desc.layout = ""

                try:
                    grid_shape = desc["layout"]["core_range_set"][0]["size"]
                    tensor_desc.grid_shape = [grid_shape["x"], grid_shape["y"]]
                except KeyError:
                    pass

                tensor_desc_list.append(tensor_desc)
        return tensor_desc_list

    def extract_program_io_details(self, program_filter: Optional[List[str]] = None):
        """
        Extracts detailed input and output configurations for each program from the flatbuffer binary JSON.

        Args:
            program_filter (Optional[List[str]]): A list of program names to filter the extraction process.
                Only programs whose names appear in this list will have their input/output details extracted.
                If None, details for all programs will be extracted.
        Returns:
            tuple: A tuple (program_inputs, program_outputs) where:
                - program_inputs (Dict[str, List[TensorDesc]]): Maps program names to detailed input configurations.
                - program_outputs (Dict[str, List[TensorDesc]]): Maps program names to detailed output configurations.
            Returns (None, None) if the binary JSON does not contain a "programs" key.
        """
        if "programs" not in self.binary:
            return None, None

        program_inputs = {}
        program_outputs = {}

        for program in self.binary["programs"]:
            program_name = program["name"]
            if program_filter is not None and program_name not in program_filter:
                continue
            inputs = self.extract_tensor_details(program["inputs"])
            outputs = self.extract_tensor_details(program["outputs"])
            if len(inputs) > 0:
                program_inputs[program_name] = inputs
            if len(outputs) > 0:
                program_outputs[program_name] = outputs

        return program_inputs, program_outputs


@dataclass_json
@dataclass
class Operand:
    node_type: str = ""
    shape: Optional[Tuple[int, ...]] = None
    dataformat: str = ""
    torch_dtype: str = ""


@dataclass_json
@dataclass
class OpInfo:
    forge_op_name: str = ""
    args: Dict[str, Any] = field(default_factory=lambda: dict())
    operands: Optional[List[Operand]] = None
    model_names: Optional[List[str]] = None


@dataclass_json
@dataclass
class Config:
    compiler: Dict[str, Any] = field(default_factory=lambda: dict())
    verify: Dict[str, Any] = field(default_factory=lambda: dict())


@dataclass_json
@dataclass
class ModelInfo:
    framework: str = ""
    model_arch: str = ""
    variant_name: str = ""
    task: str = ""
    source: str = ""


@dataclass_json
@dataclass
class Tags:
    model_name: str = ""
    bringup_status: str = ""
    execution_stage: str = ""
    pcc: Optional[float] = None
    atol: Optional[float] = None
    op_info: Optional[OpInfo] = None
    inputs: Optional[List[TensorDesc]] = None
    outputs: Optional[List[TensorDesc]] = None
    model_info: Optional[ModelInfo] = None
    failure_category: str = ""
    refined_error_message: str = ""


@dataclass_json
@dataclass
class ForgePropertyStore:
    owner: str = "tt-forge-fe"
    group: str = ""
    tags: Optional[Tags] = None
    config: Optional[Config] = None


class ForgePropertyHandler:
    """
    Handles storage and retrieval of properties in a nested ForgePropertyStore.

    This class provides methods to add, update, retrieve, and clean properties stored in a
    ForgePropertyStore. It supports nested attributes using dot-separated keys and includes
    several utility methods for recording specific property values (such as group, model name,
    and configuration data).

    Attributes:
        store (ForgePropertyStore): The underlying store containing the property data.
        record_single_op_details (bool): Flag indicating whether single operation details
        should be recorded.
    """

    def __init__(self, store: ForgePropertyStore):
        self.store = store
        self.record_single_op_details = False

    def enable_single_op_details_recording(self):
        """
        Enables recording for single operation details.

        When enabled, additional details such as op name, op arguments, operands information,
        model names, refined error messages, and failure categories will be recorded.
        """
        self.record_single_op_details = True

    def disable_single_op_details_recording(self):
        """
        Disables recording for single operation details.
        """
        self.record_single_op_details = False

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

    def record_model_name(self, model_name: str):
        """
        Records the model name in the tags.

        Args:
            model_name (str): The model name to record.
        """
        self.add("tags.model_name", model_name)

    def record_pcc(self, pcc: float):
        """
        Records the PCC metric in the tags.

        Args:
            pcc (float): PCC; correlation accuracy (measured and recorded agains compiled model)
        """
        self.add("tags.pcc", pcc)

    def record_atol(self, atol: float):
        """
        Records the atol (absolute tolerance) values in the tags.

        Args:
            atol (float): Absolute tolerance; numerical stability (measured and recorded agains compiled model)
        """
        self.add("tags.atol", atol)

    def record_pcc_and_atol(self, pcc: float, atol: float):
        """
        Records both PCC and atol values in the tags.

        Args:
            pcc (float): PCC; correlation accuracy (measured and recorded agains compiled model)
            atol (float): Absolute tolerance; numerical stability (measured and recorded agains compiled model)
        """
        self.record_pcc(pcc)
        self.record_atol(atol)

    def record_execution_depth(self, execution_depth: ExecutionDepth):
        """
        Records the execution depth (as bringup_status) in the tags.

        Args:
            execution_depth (ExecutionDepth): The execution depth value.
        """
        self.add("tags.bringup_status", ExecutionDepth.to_str(execution_depth))

    def record_execution_stage(self, execution_stage: ExecutionStage):
        """
        Records the execution stage in the tags.

        Args:
            execution_stage (ExecutionStage): The execution stage value.
        """
        self.add("tags.execution_stage", ExecutionStage.to_str(execution_stage))

    def record_execution(self, execution_depth: ExecutionDepth, execution_stage: ExecutionStage):
        """
        Records the execution depth and stage in the tags.

        Args:
            execution_depth (ExecutionDepth): The execution depth value.
            execution_stage (ExecutionStage): The execution stage value.
        """
        self.record_execution_depth(execution_depth)
        self.record_execution_stage(execution_stage)

    def record_compiler_config(self, compiler_config: CompilerConfig):
        """
        Records the compiler configuration under config.compiler.

        Args:
            compiler_config (CompilerConfig): The compiler configuration object.
        """
        self.add("config.compiler", compiler_config.to_dict())

    def record_verify_config(self, verify_config: VerifyConfig):
        """
        Records the verify configuration under config.verify.

        Converts the verify configuration to a dictionary, and ensures that the value
        for the 'value_checker' key is also represented as a dictionary.

        Args:
            verify_config (VerifyConfig): The verify configuration object.
        """
        verify_config = verify_config.to_dict()
        verify_config["value_checker"] = verify_config["value_checker"].__dict__
        self.add("config.verify", verify_config)

    def record_flatbuffer_inputs(self, inputs: List[TensorDesc]):
        """
        Records forward program inputs tensor description extracted from a flatbuffer binary.

        Args:
            inputs (List[TensorDesc]): A list of TensorDesc objects for inputs.
        """
        self.add("tags.inputs", inputs)

    def record_flatbuffer_outputs(self, outputs: List[TensorDesc]):
        """
        Records forward program outputs tensor description extracted from a flatbuffer binary.

        Args:
            outputs (List[TensorDesc]): A list of TensorDesc objects for outputs.
        """
        self.add("tags.outputs", outputs)

    def record_flatbuffer_details(self, binary_json_str: str):
        """
        Records details from a flatbuffer binary JSON string.

        This method convert provided JSON string into a dictionary, and uses the
        FlatbufferDetailsExtractor to extract details and record it.

        Args:
            binary_json_str (str): The JSON string representation of the flatbuffer binary.
        """
        binary_json_str = re.sub(r":\s*-inf\s*([,}])", r': "-inf"\1', binary_json_str)
        binary_json_str = re.sub(r":\s*inf\s*([,}])", r': "inf"\1', binary_json_str)
        binary_json = json.loads(binary_json_str)

        flatbuffer_details_extractor = FlatbufferDetailsExtractor(binary_json)
        inputs, outputs = flatbuffer_details_extractor.extract_program_io_details(program_filter=["forward"])
        if inputs is not None and outputs is not None:
            if len(inputs) != len(outputs):
                logger.error(
                    f"Mismatch in program count: inputs have {len(inputs)} programs, while outputs have {len(outputs)} programs."
                )
            if sorted(inputs.keys()) != sorted(outputs.keys()):
                logger.error(
                    f"Mismatch in program names: inputs contain {sorted(inputs.keys())}, while outputs contain {sorted(outputs.keys())}."
                )
            self.record_flatbuffer_inputs(inputs["forward"])
            self.record_flatbuffer_outputs(outputs["forward"])

    def record_forge_op_name(self, forge_op_name: str):
        """
        Records the Forge op name in the op information tags if single op details recording is enabled.

        Args:
            forge_op_name (str): The Forge operation name.
        """
        if self.record_single_op_details:
            self.add("tags.op_info.forge_op_name", forge_op_name)

    def record_forge_op_args(self, op_args: Dict[str, Any]):
        """
        Records the arguments for the Forge operation in the op information tags if single op details recording is enabled.

        Args:
            op_args (Dict[str, Any]): A dictionary of operation arguments.
        """
        if self.record_single_op_details:
            self.add("tags.op_info.args", op_args)

    def extract_node_type(self, operand):
        if isinstance(operand, forge.Parameter):
            return "Parameter"
        elif operand.is_constant():
            return "Constant"
        else:
            return "Activation"

    def record_single_op_operands_info(self, forge_module: ForgeModule, inputs: List[Tensor]):
        """
        Records details about operation operands in the op information tags if single op details recording is enabled.

        For each operand, the function records the node type, shape, dataformat,
        and the corresponding PyTorch data type.

        forge_module (List[str]): ForgeModule to extract the operands details
        inputs (List[str]): List of forge tensor inputs for the module
        """
        if self.record_single_op_details:
            assert isinstance(
                forge_module, ForgeModule
            ), f"Operands details can be extracted only from the ForgeModule but you have provided {forge_module}"
            output = forge_module(*inputs)
            assert isinstance(output, Tensor), "ForgeModule should have only one output tensor"
            if output.src_op is not None:
                assert all(
                    True if isinstance(operand, forge.Parameter) or operand.src_op is None else False
                    for operand in output.src_op.operands
                ), "ForgeModule should contains single forge op"
                operands_list = []
                for operand in output.src_op.operands:
                    node_type = self.extract_node_type(operand)
                    shape = tuple(operand.shape.get_pytorch_shape())
                    dataformat = DataFormat.to_json(operand.data_format)
                    torch_dtype = str(operand.pt_data_format)
                    operands_list.append(
                        Operand(
                            node_type=node_type,
                            shape=shape,
                            dataformat=dataformat,
                            torch_dtype=str(torch_dtype),
                        )
                    )
                self.add("tags.op_info.operands", operands_list)

    def record_op_model_names(self, model_names: List[str]):
        """
        Records the model names associated with the operation in the op information tags if single op details recording is enabled.

        Args:
            model_names (List[str]): A list of model names.
        """
        if self.record_single_op_details:
            self.add("tags.op_info.model_names", model_names)

    def record_refined_error_message(self, refined_error_message: str):
        """
        Records the refined error message in the tags if single op details recording is enabled.

        Args:
            refined_error_message (str): The refined error message string.
        """
        if self.record_single_op_details:
            self.add("tags.refined_error_message", refined_error_message)

    def record_failure_category(self, failure_category: str):
        """
        Records the failure category in the tags if single op details recording is enabled.

        Args:
            failure_category (str): The failure category string.
        """
        if self.record_single_op_details:
            self.add("tags.failure_category", failure_category)

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

    def record_model_properties(
        self,
        framework: Framework,
        model: str,
        task: Task,
        source: Source,
        variant: str = "base",
        suffix: str | None = None,
    ) -> str:
        """
        Records model properties and generates a module name and stores it.

        Args:
            framework: The framework used (e.g., pt,tf, etc.)
            model: The model name (e.g., bert)
            variant: The model variant (e.g., bert-base-uncased)
            task: The task type (e.g., qa,mlm, etc.)
            source: The model source (e.g., hf,torchhub etc.)
            suffix: Optional suffix to append to the module name

        Returns:
            The generated module name
        """

        # Record individual properties
        self.add("tags.model_info.framework", framework.full)
        self.add("tags.model_info.model_arch", model)
        self.add("tags.model_info.variant_name", variant)
        self.add("tags.model_info.task", task.full)
        self.add("tags.model_info.source", source.full)

        # Build and return the module name
        module_name = build_module_name(
            framework=framework.short, model=model, variant=variant, task=task.short, source=source.short, suffix=suffix
        )

        # Record model_name
        self.record_model_name(module_name)
        return module_name
