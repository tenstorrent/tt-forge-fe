# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Any, Tuple, Optional
from enum import IntEnum
from loguru import logger

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config

from forge._C import DataFormat, ForgeGraphModule, GraphType
from forge._C.graph import Graph, RuntimeTensorTransform
import forge._C.graph as pygraph
from forge._C.runtime import run_binary, Binary
from forge.utils import list_as_json
from forge.tensor import Tensor, get_post_const_eval_tensors, to_pt_tensors
from forge.module import Module
from forge.typing import AnyTensor, AnyModule


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
    ordered_target_names: List[str]
    ordered_constant_node_names: List[str]
    ordered_parameter_node_names: List[str]
    ordered_intermediate_names: List[str]

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
    def from_compiled_graph(module: Module, graph: Graph) -> "CompiledGraphState":
        ordered_input_names = graph.get_ordered_input_names()
        ordered_output_names = graph.get_ordered_output_names()
        ordered_target_names = graph.get_ordered_target_names()
        ordered_intermediate_names = graph.get_ordered_intermediate_names()
        ordered_constant_node_names=[constant_node.name for constant_node in graph.get_constant_nodes()]
        ordered_parameter_node_names=[parameter_node.name for parameter_node in graph.get_parameter_nodes()]

        # TODO: will be needed for training
        optimizer_param_info = {}

        consteval_trace = pygraph.record_consteval_operations(graph)
        has_cache_buffers = False

        constant_to_tensor = {}
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
            ordered_constant_node_names,
            is_forge=False
        )

        post_const_eval_parameters = {}
        post_const_eval_parameters: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph,
            constant_to_tensor,
            consteval_trace,
            ordered_parameter_node_names,
            is_forge=False
        )

        return CompiledGraphState(
            graph=graph,
            ordered_input_names=ordered_input_names,
            ordered_output_names=ordered_output_names,
            ordered_target_names=ordered_target_names,
            ordered_constant_node_names=ordered_constant_node_names,
            ordered_parameter_node_names=ordered_parameter_node_names,
            ordered_intermediate_names=ordered_intermediate_names,
            consteval_trace=consteval_trace,
            optimizer_param_info=optimizer_param_info,
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

class ProgramId(IntEnum):
    FORWARD = 0
    BACKWARD = 1

class CompiledModel:
    """
    Callable object for running inference on the compiled model.
    """
    fwd_compiled_graph_state: CompiledGraphState
    bwd_compiled_graph_state: Optional[CompiledGraphState]
    compiled_binary: Binary
    inputs: List[torch.Tensor]
    intermediates: List[torch.Tensor]
    framework_module: AnyModule
    loss_module: Optional[Module]
    optimizer: Optional[torch.optim.Optimizer]

    def __init__(self, fwd_compiled_graph_state: CompiledGraphState, bwd_compiled_graph_state: CompiledGraphState, compiled_binary: Binary, framework_module: AnyModule, loss_module: Optional[Module] = None, optimizer: Optional[torch.optim.Optimizer] = None):
        self.fwd_compiled_graph_state = fwd_compiled_graph_state
        self.bwd_compiled_graph_state = bwd_compiled_graph_state
        self.compiled_binary = compiled_binary
        self.inputs = []
        self.framework_module = framework_module
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.intermediates = []

    def __call__(self, *inputs: AnyTensor) -> List[torch.Tensor]:
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
        inputs_and_parameters = [*inputs, *self.fwd_compiled_graph_state.get_ordered_constant_tensors(), *self.fwd_compiled_graph_state.get_ordered_parameter_tensors()]

        if any([not isinstance(t, torch.Tensor) for t in inputs_and_parameters]):
            logger.info("Converting inputs and parameters to PyTorch tensors...")
            inputs_and_parameters = to_pt_tensors(inputs_and_parameters)

        logger.info(f"Running model {self.fwd_compiled_graph_state.graph.get_name()} on device...")
        model_outputs = run_binary(self.compiled_binary, int(ProgramId.FORWARD), inputs_and_parameters)

        self.intermediates = []
        for idx, output_name in enumerate(self.fwd_compiled_graph_state.ordered_output_names):
            if output_name in self.fwd_compiled_graph_state.ordered_intermediate_names:
                self.intermediates.append(model_outputs[idx])

        self.outputs = {}
        self.outputs[self.fwd_compiled_graph_state.ordered_output_names[0]] = model_outputs[0]

        model_outputs = [model_outputs[0]]
        
        if self.fwd_compiled_graph_state.graph.training():
            # For executing loss and its backward graph on CPU, we need to tell torch to compute gradients.
            for output in model_outputs:
                output.requires_grad = True

        return model_outputs

    def forward(self, *inputs: AnyTensor) -> List[torch.Tensor]:
        return self(inputs)

    def backward(self, loss_grad: torch.Tensor) -> List[torch.Tensor]:
        assert self.fwd_compiled_graph_state.graph.training(), "Model not compiled for training."
        assert self.bwd_compiled_graph_state is not None, "Backward graph should be present for training."
        consts_and_params = [*self.bwd_compiled_graph_state.get_ordered_constant_tensors(), *self.bwd_compiled_graph_state.get_ordered_parameter_tensors()]

        # Make a list from gradients passed from loss function.
        if not isinstance(loss_grad, list):
            loss_grad = [loss_grad]

        logger.info(f"Running backward pass on model {self.bwd_compiled_graph_state.graph.get_name()} on device...")
        grads = run_binary(self.compiled_binary, int(ProgramId.BACKWARD), [*loss_grad, *self.intermediates, *self.inputs, *consts_and_params])

        for name, param in self.framework_module.module.named_parameters():
            for idx, grad in enumerate(self.bwd_compiled_graph_state.ordered_output_names):
                if name in grad:
                    if (param.shape != grads[idx].shape):
                        # Our gradients for bias are 2D (of [1, N] shape) but PyTorch expects 1D - [N].
                        assert (torch.squeeze(grads[idx], 0)).shape == param.shape
                        grads[idx] = torch.squeeze(grads[idx], 0)

                    if param.grad is not None:
                        param.grad += grads[idx]
                    else:
                        param.grad = grads[idx]
            
        return grads

