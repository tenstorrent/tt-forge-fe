from . import autograd as autograd, backend_api as backend_api, balancer as balancer, graph as graph, pattern_matcher as pattern_matcher, scheduler as scheduler, torch_device as torch_device
from typing import ClassVar, Dict, List, Optional, Tuple, Union

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
HiFi2: MathFidelity
HiFi3: MathFidelity
HiFi4: MathFidelity
Int8: DataFormat
Invalid: MathFidelity
Lf8: DataFormat
LoFi: MathFidelity
Optimizer: NodeEpochType
RawUInt16: DataFormat
RawUInt32: DataFormat
RawUInt8: DataFormat
UInt16: DataFormat
VERSION: int
k_dim: int

class AMPNodeProperties:
    def __init__(self, op_type: Optional[str] = ..., epoch_type: Optional[NodeEpochType] = ..., output_df: Optional[DataFormat] = ..., intermediate_df: Optional[DataFormat] = ..., accumulate_df: Optional[DataFormat] = ..., math_fidelity: Optional[MathFidelity] = ..., name_regex_match: Optional[str] = ..., input_df: Optional[Union[Dict[int,Tuple[DataFormat,bool]],DataFormat]] = ..., is_gradient_op: Optional[bool] = ..., input_parameter_indices_to_optimize: Optional[List[Tuple[int,int]]] = ...) -> None: ...
    def from_json(self) -> AMPNodeProperties: ...
    def to_json(self) -> json: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def accumulate_df(self) -> Optional[DataFormat]: ...
    @property
    def epoch_type(self) -> Optional[NodeEpochType]: ...
    @property
    def input_df(self) -> Optional[Union[Dict[int,Tuple[DataFormat,bool]],DataFormat]]: ...
    @property
    def input_parameter_indices_to_optimize(self) -> Optional[List[Tuple[int,int]]]: ...
    @property
    def intermediate_df(self) -> Optional[DataFormat]: ...
    @property
    def is_gradient_op(self) -> Optional[bool]: ...
    @property
    def math_fidelity(self) -> Optional[MathFidelity]: ...
    @property
    def name_regex_match(self) -> Optional[str]: ...
    @property
    def op_type(self) -> Optional[str]: ...
    @property
    def output_df(self) -> Optional[DataFormat]: ...

class Block:
    def __init__(self) -> None: ...

class Blocks:
    def __init__(self) -> None: ...

class BudaNetlist:
    def __init__(self) -> None: ...
    def append_comment(self, arg0: str) -> None: ...
    def dump_to_yaml(self) -> str: ...

class BudaNetlistConfig:
    def __init__(self) -> None: ...

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
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DramQueueConfigOverride:
    def __init__(self, arg0: Optional[int], arg1: Optional[int]) -> None: ...
    def from_json(self) -> DramQueueConfigOverride: ...
    def to_json(self) -> Dict[str,Optional[int]]: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...

class InsertionInstruction:
    def __init__(self, src: str, dest: str, hoist_tms: bool, input_id: Optional[int] = ..., fork_id: Optional[int] = ..., user_defined: bool = ...) -> None: ...
    def insert(self, arg0: graph.Graph) -> None: ...
    def unique_id(self) -> Tuple[str,str,int,int,bool]: ...

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
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
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
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class NopInsertionInstruction(InsertionInstruction):
    def __init__(self, src: str, dest: str, hoist_tms: bool, nop_count: int = ..., input_id: Optional[int] = ..., fork_id: Optional[int] = ..., user_defined: bool = ..., mergeable: bool = ..., daisy_chain: bool = ..., request_merge: bool = ...) -> None: ...
    def from_json(self) -> NopInsertionInstruction: ...
    def to_json(self) -> Dict[str,Union[str,bool,int,Optional[int]]]: ...
    def unique_id(self) -> Tuple[str,str,int,int,bool]: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...

class PostPlacerConfig:
    def __init__(self, device_config: backend_api.DeviceConfig, microbatch_size: int, microbatch_count: int, enable_t_streaming: bool, input_queues_on_host: bool, output_queues_on_host: bool, manual_dram_queue_placement: Dict[str,DramQueueConfigOverride], fork_join_tiles_treshold: int, output_queue_multiplier: int, input_queue_multiplier: int, enable_cross_chip_buffering: bool, placement_algorithm: placer.DRAMPlacementAlgorithm) -> None: ...

class PostPlacerResults:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def allocated_blocks(self) -> List[List[Blocks]]: ...
    @property
    def current_host_address(self) -> int: ...
    @property
    def ins_instructions(self) -> Dict[Tuple[str,str,int,int,bool],InsertionInstruction]: ...
    @property
    def perf_model_results(self) -> Dict[str,float]: ...

class QueueInsertionInstruction(InsertionInstruction):
    def __init__(self, src: str, dest: str, hoist_tms: bool, num_entries: int, queue_size: int, input_id: Optional[int] = ..., fork_id: Optional[int] = ..., user_defined: bool = ...) -> None: ...
    def unique_id(self) -> Tuple[str,str,int,int,bool]: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...

class SparseBUDA:
    def __init__(self, *args, **kwargs) -> None: ...
    def get_sparse_tile_ptr_bits(self, arg0: int, arg1: int, arg2: int) -> int: ...
    def get_sparse_tiles_and_encodings(self, arg0: int) -> Tuple[List[List[float]],List[List[int]],List[int],List[int],List[int]]: ...
    @property
    def bcast_factor(self) -> int: ...
    @property
    def sparse_indices(self) -> Any: ...
    @property
    def sparse_shape(self) -> List[int]: ...
    @property
    def zdim(self) -> int: ...

class SparseCOO:
    def __init__(self, rows: List[int], cols: List[int], vals: List[float], shape: List[int]) -> None: ...
    @property
    def cols(self) -> List[int]: ...
    @property
    def rows(self) -> List[int]: ...
    @property
    def shape(self) -> List[int]: ...
    @property
    def vals(self) -> List[float]: ...

class UnsupportedHWOpsError(Exception): ...

def calculate_splice_buda_attrs(org_buda_attrs: List[int] = ..., splice_type: str = ..., input_shape_z: int = ..., input_shape_rt: int = ..., input_shape_ct: int = ..., dim: int = ..., ublock_order_cuts_dim: bool = ..., index: int = ..., length: int = ..., stride: int = ..., grid_r: int = ..., grid_c: int = ..., ublock_rt: int = ..., ublock_ct: int = ..., t_stream_factor_t_dim: int = ...) -> Tuple[int,int,int]: ...
def compress_sparse_tensor_and_strip_info(arg0: List[SparseCOO], arg1: int, arg2: int) -> SparseBUDA: ...
def dump_epoch_id_graphs(graph: graph.Graph, test_name: str, graph_name: str, placer_solution: placer.PlacerSolution, balancer_solution: balancer.BalancerSolution = ...) -> None: ...
def dump_epoch_type_graphs(graph: graph.Graph, test_name: str, graph_name: str, placer_solution: placer.PlacerSolution = ..., balancer_solution: balancer.BalancerSolution = ...) -> None: ...
def dump_graph(graph: graph.Graph, test_name: str, graph_name: str, placer_solution: placer.PlacerSolution = ..., balancer_solution: balancer.BalancerSolution = ...) -> None: ...
def is_subset_of_instructions(ins_instructions: Dict[Tuple[str,str,int,int,bool],InsertionInstruction] = ..., previous_instructions: Dict[Tuple[str,str,int,int,bool],InsertionInstruction] = ...) -> Tuple[bool,int,int]: ...
def link_past_cache_ios(arg0: graph.Graph) -> Dict[str,int]: ...
def lower_to_buda_netlist(arg0: graph.Graph, arg1: str, arg2: placer.PlacerSolution, arg3: balancer.BalancerSolution, arg4: List[int], arg5: backend_api.DeviceConfig) -> BudaNetlist: ...
def merge_netlists(arg0: List[BudaNetlist]) -> BudaNetlist: ...
def run_consteval_graph_pass(arg0: graph.Graph) -> None: ...
def run_optimization_graph_passes(arg0: graph.Graph, arg1: backend_api.DeviceConfig) -> None: ...
def run_placer_buda_passes(arg0: graph.Graph, arg1: balancer.BalancerConfig, arg2: Dict[str, int], arg3: dict) -> Tuple[balancer.BalancerSolution, bool]: ...
def run_post_autograd_graph_passes(arg0: graph.Graph, arg1: object) -> List[Tuple[int, int]]: ...
def run_post_initial_graph_passes(arg0: graph.Graph, arg1: object, arg2: List[Tuple[List[Tuple[str, List[int], List[int]]], Dict[str, List[int]]]]) -> Tuple[List[Tuple[int, int]], Dict[str, int]]: ...
def run_post_optimize_decompose_graph_passes(arg0: graph.Graph, arg1: object) -> List[Tuple[int, int]]: ...
def run_post_placer_buda_passes(arg0: graph.Graph, arg1: str, arg2: backend_api.DeviceConfig, arg3: placer.PlacerSolution, arg4: PostPlacerConfig, arg5: balancer.BalancerSolution, arg6: Dict[Tuple[str, str, int, int, bool], InsertionInstruction], arg7: List[List[Blocks]], arg8: int) -> PostPlacerResults: ...
def run_lower_to_mlir_passes(arg0: graph.Graph) -> None: ...
def run_pre_netlist_generation_buda_passes(arg0: graph.Graph, arg1: str, arg2: backend_api.DeviceConfig, arg3: Dict[str, object], arg4: placer.PlacerSolution, arg5: PostPlacerConfig, arg6: balancer.BalancerSolution, arg7: List[List[Blocks]], arg8: int) -> None: ...
def run_pre_placer_buda_passes(graph: graph.Graph, scheduler_config: scheduler.SchedulerConfig, device_config: backend_api.DeviceConfig, chip_ids: List[int] = ..., op_names_to_chip_break: List[List[str]] = ..., op_names_to_epoch_break: List[List[str]] = ..., op_names_dont_fuse: List[str] = ..., op_names_manual_fuse: List[str] = ..., fracture_chip_id_assignments: Dict[str, int] = ..., default_df_override: Optional[DataFormat] = ..., default_accumulate_df: Optional[DataFormat] = ..., enable_broadcast_splitting: bool = ..., fp32_fallback: DataFormat = ..., default_math_fidelity: MathFidelity = ..., enable_auto_fusing: bool = ..., amp_level: int = ..., enable_recompute: bool = ..., output_queues_on_host: bool = ..., input_queues_on_host: bool = ..., ins_instructions: Dict[Tuple[str, str, int, int, bool], InsertionInstruction] = ..., insert_queues: List[Tuple[str, str, int]] = ..., amp_properties=..., op_intermediates_to_save: List[str] = ..., use_interactive_placer: bool = ..., enable_device_tilize: bool = ...) -> Tuple[graph.Graph, placer.PlacerConfigUpdate]: ...
