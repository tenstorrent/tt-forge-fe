# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Verify by evaluating the forge graph
"""

import os
from typing import Tuple, Dict, List, Any, Union

from loguru import logger
from forge.forgeglobal import align_up_tile
import torch
import tensorflow as tf
from forge.tensor import to_pt_tensors

from ..tensor import Tensor, TensorShape, pad_pytorch_tensor_to_forge, narrow_forge_tensor_to_pytorch
from .config import DepricatedVerifyConfig, VerifyConfig, VerifyTensorMetadata, should_waive_gradient
import forge._C.graph as pygraph
from forge.tools.run_net2pipe import net2pipe
from forge.compiled_graph_state import CompiledModel
from forge.verify.compare import compare_tensor_to_golden


def _generate_random_losses(outputs, is_forge):
    losses = []
    for out in outputs:
        if out.requires_grad:
            shape = list(out.shape.get_pytorch_shape())
            if is_forge:
                while len(shape) < 4:
                    shape.insert(0, 1)
                while len(shape) > 4:
                    shape.pop(0)

                shape[-1] = align_up_tile(shape[-1])
                shape[-2] = align_up_tile(shape[-2])

            losses.append(torch.rand(shape, dtype=out.pt_data_format))
    return losses


def _run_pytorch_backward(outputs, device, losses):
    retain_graph = True
    for i, o in enumerate(outputs):
        if o.requires_grad:
            if device.loss_module is None:
                loss = narrow_forge_tensor_to_pytorch(losses[i], o.value().shape)
                o.value().backward(loss, retain_graph=retain_graph)
            else:
                o.value().backward(retain_graph=True)  # this is loss


def get_intermediate_tensors(
    graph: pygraph.Graph,
    inputs: Tuple[Tensor, ...],
    parameters: Dict[str, torch.Tensor],
    device: "TTDevice",
    is_forge: bool,
):
    torch_inputs: List[torch.Tensor] = [i.value() for i in inputs]

    if is_forge:
        torch_inputs = [
            pad_pytorch_tensor_to_forge(t, graph.get_tile_broadcast_dims_for_input(i))
            for i, t in enumerate(torch_inputs)
        ]
    intermediates = pygraph.get_intermediate_tensors(
        graph, torch_inputs, parameters, device, relative_atol=1.0, pcc=0.0
    )
    return intermediates


def do_verify(
    stage_name: str,
    training: bool,
    graph: pygraph.Graph,
    inputs: Tuple[Tensor, ...],
    parameters: Dict[str, torch.Tensor],
    golden_input_grads: Tuple[torch.Tensor, ...],
    outputs: Tuple[Tensor, ...],
    intermediate_golden_tensors: Dict,
    verify_cfg: DepricatedVerifyConfig,
    is_forge: bool,
    losses=None,
    targets: List[Tensor] = [],
    balancer_solution=None,
):
    """
    Verify graph vs. pytorch golden
    """

    torch_inputs: List[torch.Tensor] = [i.value() for i in inputs]
    torch_targets: List[torch.Tensor] = [i.value() for i in targets]

    if is_forge:
        torch_inputs = [
            pad_pytorch_tensor_to_forge(
                tensor=t,
                tile_broadcast_dims=graph.get_tile_broadcast_dims_for_input(i),
                squeeze=False,
                microbatch=1,
                tile_r=graph.get_ordered_input_tile_dims()[i][0],
                tile_c=graph.get_ordered_input_tile_dims()[i][1],
            )
            for i, t in enumerate(torch_inputs)
        ]

    if device.loss_module is not None:
        assert len(targets) > 0, f"No target provided, but device {device} has a loss module"

    logger.info("Verifying stage {}", stage_name)
    if not training:

        pcc = 0.0 if verify_cfg.pcc is None else verify_cfg.pcc
        trace_outputs, *_ = pygraph.eval(
            graph,
            torch_inputs,
            parameters,
            device,
            verify_cfg.relative_atol,
            pcc,
            intermediate_golden_tensors,
            balancer_solution=balancer_solution,
            dump_tensors_path=verify_cfg.dump_tensors_path,
            targets=torch_targets,
        )

        # Verify forward pass results
        ok = True
        for i, result in enumerate(zip(outputs, trace_outputs)):
            evaled = result[1]
            golden = result[0].value()
            ok &= compare_tensor_to_golden(f"Output {i}", golden, evaled, is_forge=is_forge, verify_cfg=verify_cfg)

    else:
        if losses is None and device.loss_module is None:
            losses = _generate_random_losses(outputs, is_forge)
        elif losses is None:
            losses = []

        # retain intermediate gradients for verification
        for t in intermediate_golden_tensors.values():
            if t.requires_grad == True:
                t.retain_grad()

        # Calculate pytorch gradients
        run_backward = False
        for i, o in enumerate(outputs):
            # Check if we need to run, or if gradients have been calculated already
            if o.value().grad is None and o.requires_grad:
                run_backward = True
                break
        if run_backward:
            _run_pytorch_backward(outputs, device, losses)

        pcc = 0.0 if verify_cfg.pcc is None else verify_cfg.pcc
        trace_outputs, parameter_to_gradients, bwd_gradients, parameter_to_updated_parameter = pygraph.eval(
            graph,
            torch_inputs,
            parameters,
            tt_device=device,
            relative_atol=verify_cfg.relative_atol,
            pcc=pcc,
            intermediate_golden_tensors=intermediate_golden_tensors,
            losses=losses,
            targets=torch_targets,
            balancer_solution=balancer_solution,
            dump_tensors_path=verify_cfg.dump_tensors_path,
        )

        # Verify forward pass results
        ok = True
        for i, result in enumerate(zip(outputs, trace_outputs)):
            evaled = result[1]
            golden = result[0].value()
            ok &= compare_tensor_to_golden(f"Output {i}", golden, evaled, is_forge=is_forge, verify_cfg=verify_cfg)

        # Verify bwd gradients
        # allow 0 on golden below because on the first post-autograd pass we don't have golden input grads yet
        assert len(golden_input_grads) == 0 or (
            len(golden_input_grads) == len(bwd_gradients)
        ), f"Golden has {len(golden_input_grads)} input gradients, but graph eval returned {len(bwd_gradients)}"
        for bwd_index, golden_input_grad in enumerate(golden_input_grads):
            evaled = bwd_gradients[bwd_index]
            ok &= compare_tensor_to_golden(
                f"Bwd gradient {bwd_index}", golden_input_grad, evaled, is_forge=is_forge, verify_cfg=verify_cfg
            )

        # Verify parameter gradients:
        device_parameters = device.get_parameters()
        for parameter in device_parameters:
            if parameter.requires_grad:
                parameter_name = parameter.get_name()
                if not parameter_name in parameter_to_gradients:
                    logger.warning("Parameter {} not used.", parameter_name)
                    continue

                golden = parameter.value().grad
                assert golden is not None
                evaled = parameter_to_gradients[parameter_name]
                warning_only = should_waive_gradient(parameter_name, verify_cfg)
                ok &= compare_tensor_to_golden(
                    f"Gradient for {parameter_name}",
                    golden,
                    evaled,
                    is_forge=is_forge,
                    verify_cfg=verify_cfg,
                    warning_only=warning_only,
                )

        # Verify parameter updates:
        optimizer = device.get_optimizer()
        if optimizer:
            for parameter in device_parameters:
                if parameter.requires_grad:
                    parameter_name = parameter.get_name()
                    if not parameter_name in parameter_to_updated_parameter:
                        logger.warning("Parameter {} not used.", parameter_name)
                        continue

                    golden = optimizer.torch_parameter_update(
                        parameter_name=parameter_name, parameter=parameter.value(), gradient=parameter.value().grad
                    )
                    evaled = parameter_to_updated_parameter[parameter_name]
                    warning_only = should_waive_gradient(parameter_name, verify_cfg)
                    ok &= compare_tensor_to_golden(
                        f"Parameter Update for {parameter_name}",
                        golden,
                        evaled,
                        is_forge=is_forge,
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
    verify_cfg: DepricatedVerifyConfig,
):

    assert False  # Run ttnn golden


def verify(
    inputs: List[Union[torch.Tensor, tf.Tensor, tf.Variable]],
    framework_model: Union[torch.nn.Module, tf.Module, tf.keras.Model],
    compiled_model: CompiledModel,
    verify_cfg: VerifyConfig = VerifyConfig(),
):
    """
    Verify the compiled model against the framework model
    """
    if not verify_cfg.enabled:
        logger.warning("Verification is disabled")
        return

    # 0th step: input checks

    # Check if inputs are of the correct type
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
    co_out = compiled_model(*inputs)

    # 2nd step: apply preprocessing (push tensors to cpu, perform any reshape if necessary,
    #  cast from tensorflow tensors to pytorch tensors if needed)
    fw_out = to_pt_tensors(fw_out)

    assert all(isinstance(co, torch.Tensor) for co in co_out), f"Compiled model output is not a list of torch.Tensor"

    co_out = [co.to("cpu") for co in co_out]

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
            if fw.dtype != co.dtype:
                raise TypeError(f"Dtype mismatch: framework_model.dtype={fw.dtype}, compiled_model.dtype={co.dtype}")

        if verify_cfg.verify_shape:
            if fw.shape != co.shape:
                raise ValueError(f"Shape mismatch: framework_model.shape={fw.shape}, compiled_model.shape={co.shape}")

        if verify_cfg.verify_values:
            verify_cfg.value_checker.check(fw, co)
