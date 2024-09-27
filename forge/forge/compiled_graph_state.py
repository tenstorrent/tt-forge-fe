# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Any, Tuple, Optional
from enum import IntEnum
from loguru import logger

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config

from forge._C import DataFormat
from forge._C.graph import Graph, RuntimeTensorTransform
from forge._C.runtime import run_binary, Binary
from forge.utils import list_as_json
from forge.tensor import Tensor, get_post_const_eval_tensors, to_pt_tensors
from forge.module import Module
from forge.typing import AnyTensor


import torch

def no_encoding(obj):
    return obj # perform json-encoding later
def no_decoding(obj):
    return obj # perform json-encoding later
def optional_no_encoding(obj):
    return None if obj is None else obj
def optional_no_decoding(obj):
    return None if obj is None else obj

class CompileResults:
    """
    Wrapper for result from the graph compiler. Contains initial and final graphs, output tensors,
    and, optionally golden results for final output and intermediates, if desired.
    """
    outputs: List[Tensor]
    golden_outputs: List[torch.Tensor]
    golden_intermediates: Dict[str, torch.Tensor]
    initial_graph: Graph
    final_graph: Graph
    loss_module: Optional[Module]
    optimizer: Optional[torch.optim.Optimizer]

    pass_specific_output_kwargs: Dict[str, Any] = {}

@dataclass_json
@dataclass()
class CompiledGraphState:
    graph: Graph
    ordered_input_names: List[str]
    ordered_output_names: List[str]
    ordered_input_gradient_names: List[str]
    ordered_output_gradient_names: List[str]
    ordered_target_names: List[str]
    ordered_constant_node_names: List[str]
    ordered_parameter_node_names: List[str]
    ordered_intermediate_activation_names: List[Tuple[str,str]]
    ordered_input_subgraph_indices: List[int]
    ordered_output_subgraph_indices: List[int]
    ordered_target_subgraph_indices: List[int]

    ordered_input_tile_broadcast_dims: List[List[int]]
    ordered_target_tile_broadcast_dims: List[List[int]]
    ordered_bw_input_tile_broadcast_dims: List[List[int]]

    ordered_input_runtime_tensor_transforms: List[RuntimeTensorTransform] = field(
        metadata=list_as_json(RuntimeTensorTransform)
    )
    ordered_output_runtime_tensor_transforms: List[RuntimeTensorTransform] = field(
        metadata=list_as_json(RuntimeTensorTransform)
    )

    input_to_tile_dims: Dict[str, Tuple[int, int]]
    parameter_to_tile_dims: Dict[str, Tuple[int, int]]
    constant_to_tile_dims: Dict[str, Tuple[int, int]]

    # attributes derived based on initial graph
    ordered_input_requires_grad: List[bool]
    ordered_output_requires_grad: List[bool]
    ordered_input_shapes: List[List[int]]
    ordered_output_shapes: List[List[int]]
    ordered_target_shapes: List[List[int]]
    ordered_intermediate_shapes: List[List[int]]
    ordered_output_data_formats: List[DataFormat] = field(metadata=list_as_json(DataFormat))

    consteval_trace: Dict[str, Dict[str, Any]]
    post_const_eval_constants: Dict[str, torch.Tensor] = field(
        metadata=config( # For serialization of CompiledGraphState cls
            encoder=no_encoding, 
            decoder=no_decoding
        )
    )
    post_const_eval_parameters: Dict[str, torch.Tensor] = field(
        metadata=config( # For serialization of CompiledGraphState cls
            encoder=no_encoding, 
            decoder=no_decoding
        )
    )
    optimizer_param_info: Dict[str, List[Tuple[str, str]]]

    # Hold cpu-evaluated outputs loaded from TTI
    cpueval_outputs: Optional[List[torch.Tensor]] = field(
        metadata=config(
            encoder=optional_no_encoding,
            decoder=optional_no_decoding
        ),
        default=None
    )

    has_cache_buffers: bool = False

    @staticmethod
    def from_compiled_graph(module: Module, compile_results: CompileResults) -> "CompiledGraphState":
        graph = compile_results.final_graph
        ordered_input_names = graph.get_ordered_input_names()
        ordered_output_names = graph.get_ordered_output_names()
        ordered_input_gradient_names = graph.get_ordered_input_gradient_names()
        ordered_output_gradient_names = graph.get_ordered_output_gradient_names()
        ordered_target_names = graph.get_ordered_target_names()
        ordered_input_subgraph_indices = graph.get_ordered_input_subgraph_indices()
        ordered_output_subgraph_indices = graph.get_ordered_output_subgraph_indices()
        ordered_target_subgraph_indices = graph.get_ordered_target_subgraph_indices()
        ordered_constant_node_names=[constant_node.name for constant_node in graph.get_constant_nodes()]
        ordered_parameter_node_names=[parameter_node.name for parameter_node in graph.get_parameter_nodes()]
        ordered_intermediate_activation_names = [(intermediate.rstrip("_intermediate_output"), intermediate) for intermediate in graph.get_ordered_intermediate_names()]

        ordered_input_tile_broadcast_dims = [graph.get_tile_broadcast_dims_for_input(i) for i in range(len(ordered_input_names))]
        ordered_target_tile_broadcast_dims = [graph.get_tile_broadcast_dims_for_target(i) for i in range(len(ordered_target_names))]
        ordered_bw_input_tile_broadcast_dims = [graph.get_tile_broadcast_dims_for_bw_input(i) for i in range(len(ordered_output_gradient_names))]

        # Tile dims
        ordered_input_tile_dims = graph.get_ordered_input_tile_dims()
        ordered_parameter_tile_dims = graph.get_ordered_parameter_tile_dims()
        ordered_constant_tile_dims = graph.get_ordered_constant_tile_dims()
        input_to_tile_dims = {}
        parameter_to_tile_dims = {}
        constant_to_tile_dims = {}
        for name, tile_dim in zip(ordered_input_names, ordered_input_tile_dims):
            input_to_tile_dims[name] = tile_dim

        for name, tile_dim in zip(ordered_parameter_node_names, ordered_parameter_tile_dims):
            parameter_to_tile_dims[name] = tile_dim

        for name, tile_dim in zip(ordered_constant_node_names, ordered_constant_tile_dims):
            constant_to_tile_dims[name] = tile_dim

        # Transforms
        ordered_input_runtime_tensor_transforms = graph.get_input_runtime_tensor_transforms()
        ordered_output_runtime_tensor_transforms = graph.get_output_runtime_tensor_transforms()
        assert len(ordered_input_runtime_tensor_transforms) == len(ordered_input_names)
        assert len(ordered_output_runtime_tensor_transforms) == len(ordered_output_names)

        ordered_input_requires_grad = compile_results.initial_graph.get_ordered_input_requires_grad()
        ordered_output_requires_grad = compile_results.initial_graph.get_ordered_output_requires_grad()
        ordered_input_shapes = compile_results.initial_graph.get_ordered_input_shapes()
        if graph.output_node_redirected():
            ordered_output_shapes = graph.get_ordered_output_shapes()
        else:
            ordered_output_shapes = compile_results.initial_graph.get_ordered_output_shapes()
        ordered_target_shapes = compile_results.initial_graph.get_ordered_target_shapes()
        ordered_intermediate_shapes = graph.get_ordered_intermediate_shapes()

        # Fetching this off the output tensors, but we could also just fetch from graph
        ordered_output_data_formats = [output_tensor.data_format for output_tensor in compile_results.outputs]

        constant_to_tensor = {}
        for name, tensor in graph.get_constant_input_runtime_tensor_transform_constants():
            constant_to_tensor[name] = tensor

        # TODO: will be needed for training
        optimizer_param_info = {}

        consteval_trace = compile_results.pass_specific_output_kwargs["consteval_trace"]
        has_cache_buffers = False

        if isinstance(module, Module):
            for p in module.get_parameters():
                value = p.value(is_forge=False)
                if value == None:
                    raise ValueError(f"Parameter {p.get_name()} has no value")
                constant_to_tensor[p.get_name()] = p.value(is_forge=False)
        elif isinstance(module, torch.fx.GraphModule):
            for name, value in module.named_parameters():
                constant_to_tensor[name] = value

        post_const_eval_constants = {}
        post_const_eval_constants: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph,
            constant_to_tensor,
            consteval_trace,
            constant_to_tile_dims,
            ordered_constant_node_names,
            is_forge=False
        )

        post_const_eval_parameters = {}
        post_const_eval_parameters: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph,
            constant_to_tensor,
            consteval_trace,
            parameter_to_tile_dims,
            ordered_parameter_node_names,
            is_forge=False
        )

        return CompiledGraphState(
            graph=graph,
            ordered_input_names=ordered_input_names,
            ordered_output_names=ordered_output_names,
            ordered_input_gradient_names=ordered_input_gradient_names,
            ordered_output_gradient_names=ordered_output_gradient_names,
            ordered_target_names=ordered_target_names,
            ordered_constant_node_names=ordered_constant_node_names,
            ordered_parameter_node_names=ordered_parameter_node_names,
            ordered_intermediate_activation_names=ordered_intermediate_activation_names,
            ordered_input_tile_broadcast_dims=ordered_input_tile_broadcast_dims,
            ordered_target_tile_broadcast_dims=ordered_target_tile_broadcast_dims,
            ordered_bw_input_tile_broadcast_dims=ordered_bw_input_tile_broadcast_dims,
            ordered_input_runtime_tensor_transforms=ordered_input_runtime_tensor_transforms,
            ordered_output_runtime_tensor_transforms=ordered_output_runtime_tensor_transforms,
            ordered_input_requires_grad=ordered_input_requires_grad,
            ordered_output_requires_grad=ordered_output_requires_grad,
            ordered_input_shapes=ordered_input_shapes,
            ordered_output_shapes=ordered_output_shapes,
            ordered_target_shapes=ordered_target_shapes,
            ordered_intermediate_shapes=ordered_intermediate_shapes,
            ordered_output_data_formats=ordered_output_data_formats,
            consteval_trace=consteval_trace,
            optimizer_param_info=optimizer_param_info,
            ordered_input_subgraph_indices=ordered_input_subgraph_indices,
            ordered_output_subgraph_indices=ordered_output_subgraph_indices,
            ordered_target_subgraph_indices=ordered_target_subgraph_indices,
            input_to_tile_dims=input_to_tile_dims,
            parameter_to_tile_dims=parameter_to_tile_dims,
            constant_to_tile_dims=constant_to_tile_dims,
            post_const_eval_constants=post_const_eval_constants,
            post_const_eval_parameters=post_const_eval_parameters,
            has_cache_buffers=has_cache_buffers,
        )

    def get_tensor(self, name_to_tensor, name):
        assert name in name_to_tensor
        value = name_to_tensor[name]

        # If mapped value is callable, we call it to get the tensor.
        # This is useful for the case where we want to lazily evaluate
        if callable(value):
            tensor = value()
            name_to_tensor[name] = tensor
        else:
            tensor = value
        return tensor

    def get_constant_tensor(self, name):
        return self.get_tensor(self.post_const_eval_constants, name)
    
    def get_ordered_constant_tensors(self):
        return [self.get_constant_tensor(name) for name in self.ordered_constant_node_names]

    def get_parameter_tensor(self, name):
        return self.get_tensor(self.post_const_eval_parameters, name)
    
    def get_ordered_parameter_tensors(self):
        return [self.get_parameter_tensor(name) for name in self.ordered_parameter_node_names]

    def get_ordered_input_names_for_subgraph(self, subgraph_idx):
        return [name for i, name in enumerate(self.ordered_input_names) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_input_shapes_for_subgraph(self, subgraph_idx):
        return [shape for i, shape in enumerate(self.ordered_input_shapes) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_input_runtime_transforms_for_subgraph(self, subgraph_idx):
        return [transform for i, transform in enumerate(self.ordered_input_runtime_tensor_transforms) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_input_tile_broadcast_dims_for_subgraph(self, subgraph_idx):
        return [tile_dims for i, tile_dims in enumerate(self.ordered_input_tile_broadcast_dims) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_output_names_for_subgraph(self, subgraph_idx):
        return [name for i, name in enumerate(self.ordered_output_names) if self.ordered_output_subgraph_indices[i] == subgraph_idx]

    def get_ordered_output_shapes_for_subgraph(self, subgraph_idx):
        return [shape for i, shape in enumerate(self.ordered_output_shapes) if self.ordered_output_subgraph_indices[i] == subgraph_idx]

    def get_ordered_output_runtime_transforms_for_subgraph(self, subgraph_idx):
        return [transform for i, transform in enumerate(self.ordered_output_runtime_tensor_transforms) if self.ordered_output_subgraph_indices[i] == subgraph_idx]

class ProgramId(IntEnum):
    FORWARD = 0
    BACKWARD = 1

class CompiledModel:
    """
    Callable object for running inference on the compiled model.
    """
    compiled_graph_state: CompiledGraphState
    compiled_binary: Binary
    inputs: List[torch.Tensor]
    loss_module: Optional[Module]
    optimizer: Optional[torch.optim.Optimizer]

    def __init__(self, compiled_graph_state: CompiledGraphState, compiled_binary: Binary, loss_module: Optional[Module] = None, optimizer: Optional[torch.optim.Optimizer] = None):
        self.compiled_graph_state = compiled_graph_state
        self.compiled_binary = compiled_binary
        self.inputs = []
        self.loss_module = loss_module
        self.optimizer = optimizer

    def __call__(self, *inputs: AnyTensor, force_dtype=None) -> List[torch.Tensor]:
        """
        Run inference on the compiled model.

        Parameters
        ----------
        inputs: [Tensor, ...]
            Input tensors

        Returns
        -------
        List[Tensor]
            Output tensors
        """
        self.inputs = [*inputs]
        inputs_and_parameters = [*inputs, *self.compiled_graph_state.get_ordered_constant_tensors(), *self.compiled_graph_state.get_ordered_parameter_tensors()]

        if any([not isinstance(t, torch.Tensor) for t in inputs_and_parameters]):
            logger.info("Converting inputs and parameters to PyTorch tensors...")
            inputs_and_parameters = to_pt_tensors(inputs_and_parameters)

        # WORKAROUND: Forcing inputs to bfloat16 for resnet because tensorflow dataformats
        # do not get propagated properly.
        if force_dtype:
            inputs_and_parameters = list(inputs_and_parameters)
            assert isinstance(force_dtype, torch.dtype), "Can only force to a pytorch dtype"
            for i in range(len(inputs_and_parameters)):
                inputs_and_parameters[i] = inputs_and_parameters[i].to(force_dtype)
            inputs_and_parameters = tuple(inputs_and_parameters)

        logger.info(f"Running model {self.compiled_graph_state.graph.get_name()} on device...")
        outputs = run_binary(self.compiled_binary, int(ProgramId.FORWARD), inputs_and_parameters)
        
        if self.compiled_graph_state.graph.training():
            # For executing loss and its backward graph on CPU, we need to tell torch to compute gradients
            for output in outputs:
                output.requires_grad = True

        return outputs

    def forward(self, *inputs: AnyTensor) -> List[torch.Tensor]:
        return self(inputs)

    def backward(self, loss_grad: torch.Tensor) -> List[torch.Tensor]:
        assert self.compiled_graph_state.graph.training(), "Model not compiled for training."

        logger.info(f"Running backward pass on model {self.compiled_graph_state.graph.get_name()} on device...")
        grads = run_binary(self.compiled_binary, int(ProgramId.BACKWARD), self.inputs + [loss_grad])
        return grads

