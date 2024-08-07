from . import autograd as autograd, graph as graph, runtime as runtime, torch_device as torch_device
from typing import ClassVar

BLACKHOLE: Arch
Backward: NodeEpochType
Bfp2: DataFormat
Bfp2_b: DataFormat
Bfp4: DataFormat
Bfp4_b: DataFormat
Bfp8: DataFormat
Bfp8_b: DataFormat
Float16: DataFormat
Float16_b: DataFormat
Float32: DataFormat
Forward: NodeEpochType
GRAYSKULL: Arch
HiFi2: MathFidelity
HiFi3: MathFidelity
HiFi4: MathFidelity
Int32: DataFormat
Int8: DataFormat
Invalid: MathFidelity
JAWBRIDGE: Arch
Lf8: DataFormat
LoFi: MathFidelity
Optimizer: NodeEpochType
RawUInt16: DataFormat
RawUInt32: DataFormat
RawUInt8: DataFormat
UInt16: DataFormat
VERSION: int
WORMHOLE: Arch
WORMHOLE_B0: Arch
k_dim: int

class AMPNodeProperties:
    def __init__(self, op_type: str | None = ..., epoch_type: NodeEpochType | None = ..., output_df: DataFormat | None = ..., intermediate_df: DataFormat | None = ..., accumulate_df: DataFormat | None = ..., math_fidelity: MathFidelity | None = ..., name_regex_match: str | None = ..., input_df: dict[int, tuple[DataFormat, bool]] | DataFormat | None | None = ..., is_gradient_op: bool | None = ..., input_parameter_indices_to_optimize: list[tuple[int, int]] | None = ...) -> None: ...
    def from_json(self) -> AMPNodeProperties: ...
    def to_json(self) -> json: ...
    @property
    def accumulate_df(self) -> DataFormat | None: ...
    @property
    def epoch_type(self) -> NodeEpochType | None: ...
    @property
    def input_df(self) -> dict[int, tuple[DataFormat, bool]] | DataFormat | None | None: ...
    @property
    def input_parameter_indices_to_optimize(self) -> list[tuple[int, int]] | None: ...
    @property
    def intermediate_df(self) -> DataFormat | None: ...
    @property
    def is_gradient_op(self) -> bool | None: ...
    @property
    def math_fidelity(self) -> MathFidelity | None: ...
    @property
    def name_regex_match(self) -> str | None: ...
    @property
    def op_type(self) -> str | None: ...
    @property
    def output_df(self) -> DataFormat | None: ...

class Arch:
    __members__: ClassVar[dict] = ...  # read-only
    BLACKHOLE: ClassVar[Arch] = ...
    GRAYSKULL: ClassVar[Arch] = ...
    Invalid: ClassVar[Arch] = ...
    JAWBRIDGE: ClassVar[Arch] = ...
    WORMHOLE: ClassVar[Arch] = ...
    WORMHOLE_B0: ClassVar[Arch] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DataFormat:
    __members__: ClassVar[dict] = ...  # read-only
    Bfp2: ClassVar[DataFormat] = ...
    Bfp2_b: ClassVar[DataFormat] = ...
    Bfp4: ClassVar[DataFormat] = ...
    Bfp4_b: ClassVar[DataFormat] = ...
    Bfp8: ClassVar[DataFormat] = ...
    Bfp8_b: ClassVar[DataFormat] = ...
    Float16: ClassVar[DataFormat] = ...
    Float16_b: ClassVar[DataFormat] = ...
    Float32: ClassVar[DataFormat] = ...
    Int32: ClassVar[DataFormat] = ...
    Int8: ClassVar[DataFormat] = ...
    Invalid: ClassVar[DataFormat] = ...
    Lf8: ClassVar[DataFormat] = ...
    RawUInt16: ClassVar[DataFormat] = ...
    RawUInt32: ClassVar[DataFormat] = ...
    RawUInt8: ClassVar[DataFormat] = ...
    UInt16: ClassVar[DataFormat] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def from_json(self) -> DataFormat: ...
    def to_json(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class MathFidelity:
    __members__: ClassVar[dict] = ...  # read-only
    HiFi2: ClassVar[MathFidelity] = ...
    HiFi3: ClassVar[MathFidelity] = ...
    HiFi4: ClassVar[MathFidelity] = ...
    Invalid: ClassVar[MathFidelity] = ...
    LoFi: ClassVar[MathFidelity] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def from_json(self) -> MathFidelity: ...
    def to_json(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class NodeEpochType:
    __members__: ClassVar[dict] = ...  # read-only
    Backward: ClassVar[NodeEpochType] = ...
    Forward: ClassVar[NodeEpochType] = ...
    Optimizer: ClassVar[NodeEpochType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SparseBUDA:
    def __init__(self, *args, **kwargs) -> None: ...
    def get_sparse_tile_ptr_bits(self, arg0: int, arg1: int, arg2: int) -> int: ...
    def get_sparse_tiles_and_encodings(self, arg0: int) -> tuple[list[list[float]], list[list[int]], list[int], list[int], list[int]]: ...
    def get_sparse_ublock_idx_bits(self, arg0: int, arg1: int, arg2: int) -> int: ...
    @property
    def bcast_factor(self) -> int: ...
    @property
    def sparse_indices(self): ...
    @property
    def sparse_shape(self) -> list[int]: ...
    @property
    def zdim(self) -> int: ...

class SparseCOO:
    def __init__(self, rows: list[int], cols: list[int], vals: list[float], shape: list[int]) -> None: ...
    @property
    def cols(self) -> list[int]: ...
    @property
    def rows(self) -> list[int]: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def vals(self) -> list[float]: ...

class UnsupportedHWOpsError(Exception): ...

def compress_sparse_tensor_and_strip_info(arg0: list[SparseCOO], arg1: int, arg2: int) -> SparseBUDA: ...
def dump_epoch_id_graphs(graph: graph.Graph, test_name: str, graph_name: str) -> None: ...
def dump_epoch_type_graphs(graph: graph.Graph, test_name: str, graph_name: str) -> None: ...
def dump_graph(graph: graph.Graph, test_name: str, graph_name: str) -> None: ...
def link_past_cache_ios(arg0: graph.Graph) -> dict[str, int]: ...
def move_index_to_mm_weights(arg0: graph.Graph) -> None: ...
def run_consteval_graph_pass(arg0: graph.Graph) -> None: ...
def run_mlir_compiler(arg0: graph.Graph) -> runtime.Binary: ...
def run_optimization_graph_passes(arg0: graph.Graph) -> None: ...
def run_post_autograd_graph_passes(arg0: graph.Graph, arg1: object) -> list[tuple[int, int]]: ...
def run_post_initial_graph_passes(arg0: graph.Graph, arg1: object, arg2: list[tuple[list[tuple[str, list[int], list[int]]], dict[str, list[int]]]]) -> tuple[list[tuple[int, int]], dict[str, int]]: ...
def run_post_optimize_decompose_graph_passes(arg0: graph.Graph, arg1: object) -> list[tuple[int, int]]: ...
def run_pre_lowering_passes(graph: graph.Graph, default_df_override: DataFormat | None = ...) -> graph.Graph: ...
def run_pre_placer_buda_passes(graph: graph.Graph, device_config, chip_ids: list[int] = ..., op_names_dont_fuse: list[str] = ..., op_names_manual_fuse: list[str] = ..., fracture_chip_id_assignments: dict[str, int] = ..., default_df_override: DataFormat | None = ..., default_accumulate_df: DataFormat | None = ..., enable_broadcast_splitting: bool = ..., fp32_fallback: DataFormat = ..., default_math_fidelity: MathFidelity = ..., enable_auto_fusing: bool = ..., amp_level: int = ..., enable_recompute: bool = ..., output_queues_on_host: bool = ..., input_queues_on_host: bool = ..., insert_queues: list[tuple[str, str, int]] = ..., amp_properties=..., op_intermediates_to_save: list[str] = ..., use_interactive_placer: bool = ..., enable_device_tilize: bool = ...) -> graph.Graph: ...
