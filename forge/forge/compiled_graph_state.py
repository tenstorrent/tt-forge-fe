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
from forge.module import Module, PyTorchModule
from forge.typing import AnyTensor, AnyModule


import torch


def no_encoding(obj):
    return obj  # perform json-encoding later


def no_decoding(obj):
    return obj  # perform json-encoding later


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
    ordered_input_gradient_names: List[str]
    ordered_output_names: List[str]
    ordered_external_output_names: List[str]
    ordered_target_names: List[str]
    ordered_constant_node_names: List[str]
    ordered_parameter_node_names: List[str]
    ordered_intermediate_names: List[str]

    consteval_trace: Dict[str, Dict[str, Any]]
    post_const_eval_constants: Dict[str, torch.Tensor] = field(
        metadata=config(encoder=no_encoding, decoder=no_decoding)  # For serialization of CompiledGraphState cls
    )
    post_const_eval_parameters: Dict[str, torch.Tensor] = field(
        metadata=config(encoder=no_encoding, decoder=no_decoding)  # For serialization of CompiledGraphState cls
    )
    optimizer_param_info: Dict[str, List[Tuple[str, str]]]

    # Hold cpu-evaluated outputs loaded from TTI
    cpueval_outputs: Optional[List[torch.Tensor]] = field(
        metadata=config(encoder=optional_no_encoding, decoder=optional_no_decoding), default=None
    )

    has_cache_buffers: bool = False

    @staticmethod
    def from_compiled_graph(module: Module, graph: Graph) -> "CompiledGraphState":
        ordered_input_names = graph.get_ordered_input_names()
        ordered_output_names = graph.get_ordered_output_names()
        ordered_external_output_names = graph.get_ordered_external_output_names()
        ordered_target_names = graph.get_ordered_target_names()
        ordered_intermediate_names = graph.get_ordered_intermediate_names()
        ordered_output_requires_grad = graph.get_ordered_output_requires_grad()
        ordered_constant_node_names = [constant_node.name for constant_node in graph.get_constant_nodes()]
        ordered_parameter_node_names = [parameter_node.name for parameter_node in graph.get_parameter_nodes()]
        ordered_input_gradient_names = graph.get_ordered_input_gradient_names()

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
            graph, constant_to_tensor, consteval_trace, ordered_constant_node_names, is_forge=False
        )

        post_const_eval_parameters = {}
        post_const_eval_parameters: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph, constant_to_tensor, consteval_trace, ordered_parameter_node_names, is_forge=False
        )

        return CompiledGraphState(
            graph=graph,
            ordered_input_names=ordered_input_names,
            ordered_input_gradient_names=ordered_input_gradient_names,
            ordered_output_names=ordered_output_names,
            ordered_external_output_names=ordered_external_output_names,
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
    Callable object for running the compiled model on the device(s).

    If the model is compiled for inference, only forward pass can be executed.
    In case of training - forward, backward, loss and optimizer steps can be executed - depending on which of these
    is compiled for the device, and which are set up to be ran separately on the CPU.
    """

    fwd_compiled_graph_state: CompiledGraphState
    bwd_compiled_graph_state: Optional[CompiledGraphState]

    # Compiled flatbuffer binary composed of programs which execute compiled graphs (e.g., forward, backward, etc.)
    compiled_binary: Binary

    inputs: List[torch.Tensor]
    outputs: Dict[str, torch.Tensor]
    intermediates: List[torch.Tensor]

    # Original user-defined module.
    framework_module: AnyModule

    # Gradients to be passed into the backward pass.
    # Used when CompiledModel.backward() is part of a chain of backward passes.
    gradient_inputs: List[torch.Tensor]
    gradient_outputs: List[torch.Tensor]

    attached_module: Optional["CompiledModel"]

    def __init__(
        self,
        fwd_compiled_graph_state: CompiledGraphState,
        bwd_compiled_graph_state: Optional[CompiledGraphState],
        compiled_binary: Binary,
        framework_module: AnyModule,
        attached_module: Optional["CompiledModel"] = None,
    ):
        self.fwd_compiled_graph_state = fwd_compiled_graph_state
        self.bwd_compiled_graph_state = bwd_compiled_graph_state
        self.compiled_binary = compiled_binary
        self.inputs = []
        self.framework_module = framework_module
        self.intermediates = []
        if self.bwd_compiled_graph_state is not None:
            assert self.bwd_compiled_graph_state is not None
            self.gradient_inputs = [None] * len(self.bwd_compiled_graph_state.ordered_input_gradient_names)
        self.outputs = {}
        self.attached_module = attached_module

    def tie_grad_fn(self, grad_id: int, grad: torch.Tensor):
        """
        Hook function to tie the gradients produced by torch as inputs to the backward pass which will be ran on the
        TT device.

        NOTE: Should be used only when loss is computed on CPU (outside of our runtime).
        """
        assert len(self.gradient_inputs) > grad_id, "More gradients than expected."
        self.gradient_inputs[grad_id] = grad

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
        inputs_and_parameters = [
            *inputs,
            *self.fwd_compiled_graph_state.get_ordered_constant_tensors(),
            *self.fwd_compiled_graph_state.get_ordered_parameter_tensors(),
        ]

        if any([not isinstance(t, torch.Tensor) for t in inputs_and_parameters]):
            logger.info("Converting inputs and parameters to PyTorch tensors...")
            inputs_and_parameters = to_pt_tensors(inputs_and_parameters)

        if self.training() and isinstance(self.framework_module, PyTorchModule):
            for name, param in self.framework_module.module.named_parameters():
                if param.requires_grad:
                    our_tensor = self.fwd_compiled_graph_state.get_parameter_tensor(name)

                    # For parameters that require gradients, we want to share the same tensor with the PyTorch module.
                    # This is done in order to be able to run optimizer step on the cpu (via torch optimizers).
                    # Ensure that this is the case:
                    assert param is our_tensor
                    assert id(param) == id(our_tensor)

        logger.info(
            f"Running model {self.framework_module.get_name()} {self.fwd_compiled_graph_state.graph.get_name()} on device..."
        )
        all_outputs = run_binary(self.compiled_binary, int(ProgramId.FORWARD), inputs_and_parameters)

        self.intermediates = []

        # The model_outputs will contain outputs that we need to return to the user, i.e. external outputs.
        model_outputs = []
        for idx, output_name in enumerate(self.fwd_compiled_graph_state.ordered_output_names):
            output = all_outputs[idx]
            if output_name in self.fwd_compiled_graph_state.ordered_intermediate_names:
                self.intermediates.append(output)
            if output_name in self.fwd_compiled_graph_state.ordered_external_output_names:
                self.outputs[output_name] = output
                model_outputs.append(output)

        if self.training():
            # For executing loss and its backward graph on CPU, we need to tell torch to compute gradients.
            for idx, output in enumerate(model_outputs):
                output.requires_grad = True
                output.register_hook(lambda grad: self.tie_grad_fn(idx, grad))

        return model_outputs

    def forward(self, *inputs: AnyTensor) -> List[torch.Tensor]:
        return self(inputs)

    def backward(self) -> List[torch.Tensor]:
        assert self.training(), "Model not compiled for training."
        assert self.bwd_compiled_graph_state is not None, "Backward graph should be present for training."
        consts_and_params = [
            *self.bwd_compiled_graph_state.get_ordered_constant_tensors(),
            *self.bwd_compiled_graph_state.get_ordered_parameter_tensors(),
        ]

        for grad in self.gradient_inputs:
            assert grad is not None, "Gradients not provided for backward pass."

        # Inputs from forward pass are needed in backward pass only if
        # they are used in the backward pass computation
        # They will be used if there is backward operation that explicitly requires them
        # as in other cases, intermediate tensors can be used if they exists
        inputs = [
            self.inputs[i]
            for i, name in enumerate(self.fwd_compiled_graph_state.ordered_input_names)
            if name in self.bwd_compiled_graph_state.ordered_input_names
        ]

        logger.info(
            f"Running backward pass on model {self.framework_module.get_name()} {self.bwd_compiled_graph_state.graph.get_name()} on device..."
        )
        grads = run_binary(
            self.compiled_binary,
            int(ProgramId.BACKWARD),
            [*self.gradient_inputs, *self.intermediates, *inputs, *consts_and_params],
        )

        # Accumulate gradients in the PyTorch module
        if isinstance(self.framework_module, PyTorchModule):
            for name, param in self.framework_module.module.named_parameters():
                for idx, grad in enumerate(self.bwd_compiled_graph_state.ordered_output_names):
                    if name in grad:
                        if param.shape != grads[idx].shape:
                            # Our gradients for bias are 2D (of [1, N] shape) but PyTorch expects 1D - [N].
                            assert (torch.squeeze(grads[idx], 0)).shape == param.shape
                            grads[idx] = torch.squeeze(grads[idx], 0)

                        if param.grad is not None:
                            param.grad += grads[idx]
                        else:
                            param.grad = grads[idx]

        # Pass on the calculated gradients to the attached module
        self.gradient_outputs = grads
        if self.attached_module is not None:
            # pass on the calculated gradients and call the attached module's backward pass
            # HACK: we don't have a way to know which gradient outputs are tied to which gradient inputs
            # of the attached module. For now, just attach the first one since we are doing this only for
            # the loss module (which will have only one gradient output) and the model will need only one
            # gradient output to be passed to the loss module.
            assert len(self.gradient_outputs) == 1, "Passing gradients not properly implemented yet"
            assert len(self.attached_module.gradient_inputs) == 1, "Passing gradients not properly implemented yet"
            self.attached_module.gradient_inputs[0] = self.gradient_outputs[0]
            self.attached_module.backward()

        return grads

    def training(self) -> bool:
        return self.fwd_compiled_graph_state.graph.training()
