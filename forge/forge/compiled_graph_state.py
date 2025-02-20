# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from loguru import logger
import torch
from typing import Dict, List, Any, Optional


from forge._C.graph import Graph
import forge._C.graph as pygraph
from forge._C.runtime import run_binary, Binary
from forge.tensor import Tensor, get_post_const_eval_tensors, to_pt_tensors, AnyTensor
from forge.module import Module, PyTorchModule, AnyModule
from forge.execution_tracker import ExecutionPhase, record_execution_phase_and_stage


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
    aliased_outputs: Dict[str, str]

    # JSON trace for each node in the graph which can be consteval'd
    consteval_trace: Dict[str, Any]
    post_const_eval_constants: Dict[str, torch.Tensor]
    post_const_eval_parameters: Dict[str, torch.Tensor]

    # Hold cpu-evaluated outputs loaded from TTI
    cpueval_outputs: Optional[List[torch.Tensor]] = field(default=None)

    has_cache_buffers: bool = False

    @staticmethod
    def from_compiled_graph(
        module: Module, graph: Graph, optimizer_params: Optional[Dict[str, Tensor]] = None
    ) -> "CompiledGraphState":
        ordered_input_names = graph.get_ordered_input_names()
        ordered_output_names = graph.get_ordered_output_names()
        ordered_external_output_names = graph.get_ordered_external_output_names()
        ordered_target_names = graph.get_ordered_target_names()
        ordered_intermediate_names = graph.get_ordered_intermediate_names()
        ordered_output_nodes = graph.get_ordered_output_nodes()
        aliased_outputs: Dict[str, str] = {}
        for node in ordered_output_nodes:
            assert isinstance(node, pygraph.OutputNode)
            if node.is_aliased:
                aliased_outputs[node.name] = node.alias

        ordered_constant_node_names = [constant_node.name for constant_node in graph.get_constant_nodes()]
        ordered_parameter_node_names = [parameter_node.name for parameter_node in graph.get_parameter_nodes()]
        ordered_optimizer_parameter_node_names = [
            parameter_node.name for parameter_node in graph.get_optimizer_parameter_nodes()
        ]
        if len(ordered_optimizer_parameter_node_names) > 0:
            ordered_parameter_node_names.extend(ordered_optimizer_parameter_node_names)
        ordered_input_gradient_names = graph.get_ordered_input_gradient_names()

        consteval_trace = pygraph.record_consteval_operations(graph)

        has_cache_buffers = False

        constant_to_tensor: Dict[str, torch.Tensor] = {}
        if isinstance(module, Module):
            for p in module.get_parameters():
                value = p.value(is_forge=False)
                if value == None:
                    raise ValueError(f"Parameter {p.get_name()} has no value")
                constant_to_tensor[p.get_name()] = p.value(is_forge=False)

        if optimizer_params is not None:
            for name, opt_param in optimizer_params.items():
                constant_to_tensor[name] = opt_param.value()

        post_const_eval_constants: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph, constant_to_tensor, consteval_trace, ordered_constant_node_names, is_forge=False
        )

        post_const_eval_parameters: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph, constant_to_tensor, consteval_trace, ordered_parameter_node_names, is_forge=False
        )

        return CompiledGraphState(
            graph=graph,
            ordered_input_names=ordered_input_names,
            ordered_input_gradient_names=ordered_input_gradient_names,
            ordered_output_names=ordered_output_names,
            aliased_outputs=aliased_outputs,
            ordered_external_output_names=ordered_external_output_names,
            ordered_target_names=ordered_target_names,
            ordered_constant_node_names=ordered_constant_node_names,
            ordered_parameter_node_names=ordered_parameter_node_names,
            ordered_intermediate_names=ordered_intermediate_names,
            consteval_trace=consteval_trace,
            post_const_eval_constants=post_const_eval_constants,
            post_const_eval_parameters=post_const_eval_parameters,
            has_cache_buffers=has_cache_buffers,
        )

    def get_tensor(self, name_to_tensor: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        assert name in name_to_tensor
        return name_to_tensor[name]

    def get_constant_tensor(self, name: str) -> torch.Tensor:
        return self.get_tensor(self.post_const_eval_constants, name)

    def get_ordered_constant_tensors(self) -> List[torch.Tensor]:
        return [self.get_constant_tensor(name) for name in self.ordered_constant_node_names]

    def get_parameter_tensor(self, name: str) -> torch.Tensor:
        return self.get_tensor(self.post_const_eval_parameters, name)

    def get_ordered_parameter_tensors(self) -> List[torch.Tensor]:
        return [self.get_parameter_tensor(name) for name in self.ordered_parameter_node_names]


class ProgramId(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    OPTIMIZER = 2


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
    gradient_inputs: List[Optional[torch.Tensor]]
    gradient_outputs: List[torch.Tensor]

    attached_module: Optional["CompiledModel"]

    def __init__(
        self,
        fwd_compiled_graph_state: CompiledGraphState,
        bwd_compiled_graph_state: Optional[CompiledGraphState],
        opt_compiled_graph_state: Optional[CompiledGraphState],
        compiled_binary: Binary,
        framework_module: AnyModule,
        attached_module: Optional["CompiledModel"] = None,
    ):
        self.fwd_compiled_graph_state = fwd_compiled_graph_state
        self.bwd_compiled_graph_state = bwd_compiled_graph_state
        self.opt_compiled_graph_state = opt_compiled_graph_state
        self.compiled_binary = compiled_binary
        self.inputs = []
        self.framework_module = framework_module
        self.intermediates = []
        if self.bwd_compiled_graph_state is not None:
            self.gradient_inputs = [None] * len(self.bwd_compiled_graph_state.ordered_input_gradient_names)
        self.outputs = {}
        self.attached_module = attached_module
        self.gradient_outputs = []

    def tie_grad_fn(self, grad_id: int, grad: torch.Tensor) -> None:
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
        self.inputs = [*to_pt_tensors(inputs)]

        inputs_and_parameters = [
            *self.inputs,
            *self.fwd_compiled_graph_state.get_ordered_constant_tensors(),
            *self.fwd_compiled_graph_state.get_ordered_parameter_tensors(),
        ]

        assert all(
            [isinstance(t, torch.Tensor) for t in inputs_and_parameters]
        ), "All inputs should be torch tensors by now."

        if self.training() and isinstance(self.framework_module, PyTorchModule):
            for name, param in self.framework_module.module.named_parameters():
                if param.requires_grad:
                    our_tensor = self.fwd_compiled_graph_state.get_parameter_tensor(name)

                    # NOTE: for parameters that require gradients, we want to share the same tensor with the PyTorch
                    # module. This is because we want to be able to optimize the parameters both on the device
                    # (through our runtime) and via the torch optimizers. So this ensures that whichever side updates
                    # the parameter value, the other side can see the change.
                    #
                    # This could change in the future, but for now ensure that our premise is correct.
                    assert param is our_tensor

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
                # NOTE: the default idx parameter for the lambda is used to capture the idx by value. Otherwise, the lambda
                # would capture the idx by reference, and all the lambdas would have the same idx value.
                output.register_hook(lambda grad, idx=idx: self.tie_grad_fn(idx, grad))

        record_execution_phase_and_stage(ExecutionPhase.EXECUTED_TTNN)

        return model_outputs

    def forward(self, *inputs: AnyTensor) -> List[torch.Tensor]:
        return self(*inputs)

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

        if self.optimizer_on_device():
            if self.gradient_outputs is None or len(self.gradient_outputs) == 0:
                self.gradient_outputs = grads
            else:
                assert len(self.gradient_outputs) == len(grads), "Number of gradients does not match number of outputs"
                for idx, grad in enumerate(grads):
                    self.gradient_outputs[idx] += grad
        else:
            self.gradient_outputs = grads
            # Accumulate gradients in the PyTorch module
            if isinstance(self.framework_module, PyTorchModule):
                for name, param in self.framework_module.module.named_parameters():
                    for idx, grad_output_name in enumerate(self.bwd_compiled_graph_state.ordered_output_names):
                        if name in grad_output_name:
                            grad_tensor = grads[idx]
                            if param.shape != grad_tensor.shape:
                                # Our gradients for bias are 2D (of [1, N] shape) but PyTorch expects 1D - [N].
                                assert (torch.squeeze(grad_tensor, 0)).shape == param.shape
                                grad_tensor = torch.squeeze(grad_tensor, 0)

                            if param.grad is not None:
                                param.grad += grad_tensor
                            else:
                                param.grad = grad_tensor

        # Pass on the calculated gradients to the attached module
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

        return self.gradient_outputs

    def training(self) -> bool:
        return self.fwd_compiled_graph_state.graph.training()

    def optimizer_on_device(self) -> bool:
        return self.opt_compiled_graph_state is not None

    def step(self) -> None:
        assert self.fwd_compiled_graph_state.graph.training(), "Model not compiled for training."
        assert self.opt_compiled_graph_state is not None, "Optimizer graph should be present for training."

        inputs_and_parameters = [
            *self.gradient_outputs,
            *self.opt_compiled_graph_state.get_ordered_constant_tensors(),
            *self.opt_compiled_graph_state.get_ordered_parameter_tensors(),
        ]

        logger.info(
            f"Running optimizer step on model {self.framework_module.get_name()} {self.opt_compiled_graph_state.graph.get_name()} on device..."
        )

        out_params = run_binary(self.compiled_binary, int(ProgramId.OPTIMIZER), inputs_and_parameters)

        update_param: Dict[str, torch.Tensor] = {}
        for idx, param in enumerate(self.opt_compiled_graph_state.ordered_output_names):
            update_param[param] = out_params[idx]

        for weight_update_name in self.opt_compiled_graph_state.aliased_outputs:
            weight_name = self.opt_compiled_graph_state.aliased_outputs[weight_update_name]
            assert self.opt_compiled_graph_state.get_parameter_tensor(
                weight_name
            ) is self.fwd_compiled_graph_state.get_parameter_tensor(weight_name)
            self.fwd_compiled_graph_state.get_parameter_tensor(weight_name).data = update_param[weight_update_name].data

            # Sanity check - assert that the parameter tensors in framework module are the same as the ones in our runtime.
            assert isinstance(
                self.framework_module, PyTorchModule
            ), "For now only PyTorchModule is supported in training"
            for torch_name, val in self.framework_module.module.named_parameters():
                if torch_name == weight_name:
                    assert self.fwd_compiled_graph_state.get_parameter_tensor(weight_name) is val

        self.gradient_outputs = []
