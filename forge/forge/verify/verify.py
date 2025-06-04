# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Verify by evaluating the forge graph
"""

import os
from typing import Tuple, Dict, List, Any, Union, Optional

from forge.module import FrameworkModule
from loguru import logger
from forge.forgeglobal import align_up_tile

import torch
from forge.tensor import to_pt_tensors

from ..tensor import (
    FrameworkTensor,
    Tensor,
    narrow_forge_tensor_to_pytorch,
    pytorch_dtype_to_forge_dataformat,
    forge_dataformat_to_pytorch_dtype,
)
from .config import DeprecatedVerifyConfig, VerifyConfig, should_waive_gradient
import forge._C.graph as pygraph
from forge._C.runtime import Tensor as CTensor
from forge.compiled_graph_state import CompiledModel
from forge.verify.compare import compare_tensor_to_golden, determine_consistency_limits
from forge.verify.utils import convert_to_supported_pytorch_dtype
from forge.forge_property_utils import (
    ExecutionStage,
    record_execution,
    record_verify_config,
    record_consistency_limits,
)


def _generate_random_losses(outputs):
    losses = []
    for out in outputs:
        if out.requires_grad:
            shape = list(out.shape.get_pytorch_shape())
            losses.append(torch.rand(shape, dtype=out.pt_data_format))
    return losses


def _run_pytorch_backward(outputs, losses):
    retain_graph = True
    for i, o in enumerate(outputs):
        if o.requires_grad:
            loss = narrow_forge_tensor_to_pytorch(losses[i], o.value().shape)
            o.value().backward(loss, retain_graph=retain_graph)


def get_intermediate_tensors(
    graph: pygraph.Graph,
    inputs: Tuple[Tensor, ...],
    parameters: Dict[str, torch.Tensor],
    device: "TTDevice",
    is_forge: bool,
):
    torch_inputs: List[torch.Tensor] = [i.value() for i in inputs]
    intermediates = pygraph.get_intermediate_tensors(
        graph, torch_inputs, parameters, device, relative_atol=1.0, pcc=0.0
    )
    return intermediates


def do_verify(
    stage_name: str,
    graph: pygraph.Graph,
    inputs: Tuple[Tensor, ...],
    parameters: Dict[str, torch.Tensor],
    golden_input_grads: Tuple[torch.Tensor, ...],
    outputs: Tuple[Tensor, ...],
    intermediate_golden_tensors: Dict,
    verify_cfg: DeprecatedVerifyConfig,
    is_forge: bool,
    losses=None,
    targets: List[Tensor] = [],
    optimizer=None,
):
    """
    Verify graph vs. pytorch golden
    """
    torch_inputs: List[torch.Tensor] = [i if isinstance(i, torch.Tensor) else i.to_pytorch() for i in inputs]
    torch_targets: List[torch.Tensor] = [i if isinstance(i, torch.Tensor) else i.to_pytorch() for i in targets]

    logger.info("Verifying stage {}", stage_name)
    training = graph.contains_bwd_nodes()
    if not training:

        pcc = 0.0 if verify_cfg.pcc is None else verify_cfg.pcc
        trace_outputs, *_ = pygraph.eval(
            graph,
            torch_inputs,
            parameters,
            verify_cfg.relative_atol,
            pcc,
            intermediate_golden_tensors,
            dump_tensors_path=verify_cfg.dump_tensors_path,
            targets=torch_targets,
        )

        # Verify forward pass results
        ok = True
        for i, result in enumerate(zip(outputs, trace_outputs)):
            evaled = result[1]
            golden = result[0].value()
            ok &= compare_tensor_to_golden(f"output {i}", golden, evaled, verify_cfg=verify_cfg)

    else:
        if losses is None:
            losses = _generate_random_losses(outputs)

        # retain intermediate gradients for verification
        for t in intermediate_golden_tensors.values():
            if t.requires_grad == True:
                t.retain_grad()

        pcc = 0.0 if verify_cfg.pcc is None else verify_cfg.pcc
        trace_outputs, parameter_to_gradients, bwd_gradients, parameter_to_updated_parameter = pygraph.eval(
            graph,
            torch_inputs,
            parameters,
            relative_atol=verify_cfg.relative_atol,
            pcc=pcc,
            intermediate_golden_tensors=intermediate_golden_tensors,
            losses=losses,
            targets=torch_targets,
            dump_tensors_path=verify_cfg.dump_tensors_path,
            optimizer=optimizer,
        )

        # Verify forward pass results
        ok = True
        for i, result in enumerate(zip(outputs, trace_outputs)):
            evaled = result[1]
            golden = result[0].value()
            ok &= compare_tensor_to_golden(f"Output {i}", golden, evaled, verify_cfg=verify_cfg)

        # Verify gradients of inputs (if input tensors require gradients)
        # TODO: Here, we rely on the fact that golden inputs are ordered in the same way as in the graph.
        # This is true for now, but it is bad to rely on.
        logger.debug("Verify gradients of inputs")
        assert len(golden_input_grads) == len(
            bwd_gradients
        ), f"Golden has {len(golden_input_grads)} input gradients, but graph eval returned {len(bwd_gradients)}"
        for bwd_index, golden_input_grad in enumerate(golden_input_grads):
            evaled = bwd_gradients[bwd_index]
            ok &= compare_tensor_to_golden(
                f"gradient of input {bwd_index}", golden_input_grad, evaled, verify_cfg=verify_cfg
            )

        logger.debug("Verify gradients of parameters")
        for parameter_name, parameter_tensor in parameters.items():
            if parameter_tensor.requires_grad:
                if not parameter_name in parameter_to_gradients:
                    logger.warning("Parameter {} not used.", parameter_name)
                    continue

                golden = parameter_tensor.grad
                assert golden is not None
                evaled = parameter_to_gradients[parameter_name]
                warning_only = should_waive_gradient(parameter_name, verify_cfg)
                ok &= compare_tensor_to_golden(
                    f"gradient of parameter {parameter_name}",
                    golden,
                    evaled,
                    verify_cfg=verify_cfg,
                    warning_only=warning_only,
                )

        # Verify parameter updates if optimizer is not None:
        if optimizer:
            logger.debug("Verify updates of parameters")
            for parameter_name, parameter_tensor in parameters.items():
                if parameter_tensor.requires_grad:
                    if not parameter_name in parameter_to_updated_parameter:
                        logger.warning("Parameter {} not used.", parameter_name)
                        continue

                    golden = optimizer.torch_parameter_update(
                        parameter_name=parameter_name, parameter=parameter_tensor, gradient=parameter_tensor.grad
                    )
                    evaled = parameter_to_updated_parameter[parameter_name]
                    warning_only = should_waive_gradient(parameter_name, verify_cfg)
                    ok &= compare_tensor_to_golden(
                        f"Parameter Update for {parameter_name}",
                        golden,
                        evaled,
                        verify_cfg=verify_cfg,
                        warning_only=warning_only,
                    )

    msg = f"Stage {stage_name}: Data mismatch detected"
    if not ok:
        logger.error(msg)

    continue_on_mismatch = bool(int(os.environ.get("FORGE_CONTINUE_ON_MISMATCH", "0")))
    if not continue_on_mismatch:
        assert ok, msg
    return losses


# Make sure to clean up after ourselves, even if an abort happens
def atexit_handler(backend_api):
    backend_api.shutdown()


def verify_golden(
    netlist_filename: str,
    training: bool,
    compile_results: Any,
    device: "TTDevice",
    inputs: Tuple[Tensor],
    outputs: Tuple[torch.Tensor],
    verify_cfg: DeprecatedVerifyConfig,
):

    assert False  # Run ttnn golden


def check_dtypes(fw_dtype: torch.dtype, co_dtype: torch.dtype):
    """
    Verifies that two PyTorch data types are equivalent when considering Forge's supported data types.

    This function addresses the fact that Forge tensors support a subset of PyTorch's data types.
    For example, Forge might map torch.int64 to torch.int32 internally. When comparing outputs
    between the original PyTorch model and the Forge-compiled model, we need to account for
    these conversions to avoid false verification failures.

    The verification works by converting the framework dtype to its Forge representation
    and then back to PyTorch, then comparing this "round-trip" dtype with the compiled model's dtype.

    Args:
        fw_dtype (torch.dtype): Data type from the original PyTorch model
        co_dtype (torch.dtype): Data type from the Forge-compiled model

    Raises:
        ValueError: If the dtypes are incompatible after accounting for Forge's conversions
    """
    # Convert framework dtype to Forge's internal representation
    forge_dataformat = pytorch_dtype_to_forge_dataformat(fw_dtype)

    # Convert back to PyTorch dtype (this accounts for Forge's supported types)
    equivalent_pytorch_dtype = forge_dataformat_to_pytorch_dtype(forge_dataformat)

    # Check if the compiled dtype matches the equivalent dtype
    if equivalent_pytorch_dtype != co_dtype:
        raise ValueError(f"Dtype mismatch: framework_model.dtype={fw_dtype}, compiled_model.dtype={co_dtype}")


def verify_backward(
    inputs: List[torch.Tensor],
    output_grad: torch.Tensor,
    framework_output: torch.Tensor,
    compiled_output: torch.Tensor,
    framework_model: torch.nn.Module,
    compiled_model: CompiledModel,
    verify_cfg: VerifyConfig = VerifyConfig(),
):
    """
    Performs verification of a compiled model by comparing its outputs against a reference framework model.

    Runs backward on both models with the same inputs and performs various validation checks
    based on the provided verification configuration. Checks can include output size matching,
    dtype consistency, shape equivalence, and numeric value comparison.

    Parameters:
        inputs: List of tensor inputs
        output_grad: Output gradient tensor
        framework_output: Output tensor from the reference framework model
        compiled_output: Output tensor from the compiled model
        framework_model: Reference model
        compiled_model: compiled model to verify
        verify_cfg: Configuration object controlling which verification checks to perform
    """

    if not verify_cfg.enabled:
        logger.warning("Verification is disabled")
        return

    assert compiled_model.training(), "Compiled model must be in compiled for training for backward verification"

    # Check if inputs are of the correct type
    if not inputs:
        raise ValueError("Input tensors must be provided")

    if not isinstance(output_grad, torch.Tensor):
        raise TypeError(f"Output gradient tensor must be of type {torch.Tensor}, but got {type(output_grad)}")

    if not isinstance(framework_output, torch.Tensor):
        raise TypeError(f"Framework output tensor must be of type {torch.Tensor}, but got {type(framework_output)}")
    if not isinstance(compiled_output, torch.Tensor):
        raise TypeError(f"Compiled output tensor must be of type {torch.Tensor}, but got {type(compiled_output)}")

    if not isinstance(framework_model, torch.nn.Module):
        raise TypeError(f"Framework model must be of type {torch.nn.Module}, but got {type(framework_model)}")
    if not isinstance(compiled_model, verify_cfg.compiled_model_types):
        raise TypeError(
            f"Compiled model must be of type {verify_cfg.compiled_model_types}, but got {type(compiled_model)}"
        )

    # Zero out gradients
    [input.grad.zero_() for input in inputs if input.grad is not None]
    framework_model.zero_grad()

    # 1st step: run backward pass for the networks and get gradients
    compiled_model.gradient_inputs = [CTensor(output_grad)]
    co_gradient_outputs = compiled_model.backward()
    co_gradients: Dict[str, torch.Tensor] = {}
    for name, grad in zip(compiled_model.bwd_compiled_graph_state.ordered_output_names, co_gradient_outputs):
        # NOTE: Need to clone the gradients of parameters as they are modified in the backward pass of the framework model
        #       but no need to clone the gradients of the inputs as they are not modified in the backward pass of the framework model
        co_gradients[name] = grad.to_torch().clone() if name.startswith("grad_acc_") else grad.to_torch()

    # Run backward on framework model
    framework_model.zero_grad()
    framework_output.backward(gradient=output_grad)

    # 2nd step: verify gradients
    for name in co_gradients:
        co_grad = co_gradients[name]

        if name.startswith("grad_acc_"):
            name = name.replace("grad_acc_", "")
            name = name.replace("_grad_accumulator", "")
            fw_grad = framework_model.get_parameter(name).grad
        elif name.startswith("output_grad_"):
            name = name.replace("output_grad_", "")
            fw_grad = inputs[compiled_model.fwd_compiled_graph_state.ordered_input_names.index(name)].grad
        else:
            raise ValueError(f"Unknown gradient name in compiled model: {name}")

        assert co_grad.data_ptr() != fw_grad.data_ptr(), "Gradients are the same object in memory"

        co = co_grad.squeeze()
        fw = fw_grad.squeeze()

        if verify_cfg.verify_dtype:
            check_dtypes(fw_dtype=fw.dtype, co_dtype=co.dtype)

        if verify_cfg.verify_shape and fw.shape != co.shape:
            raise TypeError(f"Shape mismatch: framework_model.shape={fw.shape}, compiled_model.shape={co.shape}")

        if verify_cfg.verify_values:
            verify_cfg.value_checker.check(fw, co)


def verify(
    inputs: List[FrameworkTensor],
    framework_model: FrameworkModule,
    compiled_model: CompiledModel,
    verify_cfg: VerifyConfig = VerifyConfig(),
):
    """
    Performs verification of a compiled model by comparing its outputs against a reference framework model.

    Runs inference on both models with the same inputs and performs various validation checks
    based on the provided verification configuration. Checks can include output size matching,
    dtype consistency, shape equivalence, and numeric value comparison.

    Parameters:
        inputs: List of tensor inputs
        framework_model: Reference model
        compiled_model: compiled model to verify
        verify_cfg: Configuration object controlling which verification checks to perform

    Returns:
        tuple: (framework_outputs, compiled_outputs) - outputs from both models
               Returns (None, None) if verification is disabled
    """

    record_verify_config(verify_cfg)

    # 0th step: Check if inputs are of the correct type
    if not inputs:
        raise ValueError("Input tensors must be provided")

    for input_tensor in inputs:
        if not isinstance(input_tensor, verify_cfg.supported_tensor_types):
            raise TypeError(
                f"Input tensor must be of type {verify_cfg.supported_tensor_types}, but got {type(input_tensor)}"
            )

    if not isinstance(framework_model, verify_cfg.framework_model_types):
        raise TypeError(
            f"Framework model must be of type {verify_cfg.framework_model_types}, but got {type(framework_model)}"
        )

    if not isinstance(compiled_model, verify_cfg.compiled_model_types):
        raise TypeError(
            f"Compiled model must be of type {verify_cfg.compiled_model_types}, but got {type(compiled_model)}"
        )

    # 1st step: run forward pass for the networks
    fw_out = framework_model(*inputs)

    record_execution(ExecutionStage.FAILED_TTNN_BINARY_EXECUTION)
    co_out = compiled_model(*inputs)
    record_execution(ExecutionStage.FAILED_VERIFICATION)

    # 2nd step: apply preprocessing:
    # - cast framework tensors to pytorch tensors if needed
    # - convert to dtypes that are supported by our hardware
    # - push tensors to cpu, perform any reshape if necessary,
    fw_out = to_pt_tensors(fw_out)
    fw_out = tuple(convert_to_supported_pytorch_dtype(o) for o in fw_out)

    assert all(isinstance(co, torch.Tensor) for co in co_out), f"Compiled model output is not a list of torch.Tensor"

    co_out = [co.to("cpu") for co in co_out]

    record_consistency_limits(fw_out, co_out)

    if not verify_cfg.enabled:
        logger.warning("Verification is disabled")
        return fw_out, co_out

    # 3rd step: verifications of outputs
    # - size check
    # - dtype check
    # - shape check
    # - compare with golden
    if verify_cfg.verify_size:
        if len(fw_out) != len(co_out):
            raise ValueError(
                f"Number of outputs from framework model and compiled model do not match: framework model has {len(fw_out)} outputs, compiled model has {len(co_out)} outputs"
            )

    for fw, co in zip(fw_out, co_out):
        if verify_cfg.verify_dtype:
            check_dtypes(fw_dtype=fw.dtype, co_dtype=co.dtype)

        if verify_cfg.verify_shape:
            if fw.shape != co.shape:
                raise ValueError(f"Shape mismatch: framework_model.shape={fw.shape}, compiled_model.shape={co.shape}")

        if verify_cfg.verify_values:
            verify_cfg.value_checker.check(fw, co)

    record_execution(ExecutionStage.PASSED)

    # Return both the framework and compiled model outputs
    return fw_out, co_out
