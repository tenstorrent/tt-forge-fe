# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from sys import intern
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field

import torch
from loguru import logger

from .ttdevice import TTDevice
from .tensor import Tensor, pad_pytorch_tensor_to_buda
from pybuda._C import (
    link_past_cache_ios,
    move_index_to_mm_weights,
    run_post_initial_graph_passes,
    run_optimization_graph_passes,
    run_post_optimize_decompose_graph_passes,
    run_consteval_graph_pass,
    run_post_autograd_graph_passes,
    run_pre_placer_buda_passes,
    run_lower_to_mlir_passes,
    dump_graph,
    dump_epoch_type_graphs,
    dump_epoch_id_graphs,
    UnsupportedHWOpsError,
)
import pybuda
from .parameter import Parameter
import pybuda._C.autograd as pyautograd
from pybuda._C.graph import Graph
import pybuda.query as query
from .verify import VerifyConfig, do_verify, verify_golden, _generate_random_losses, _run_pytorch_backward, get_intermediate_tensors
import pybuda._C.graph as pygraph
from .config import (
    CompilerConfig,
    CompileDepth,
    _get_global_compiler_config,
)
from .pybudaglobal import state_changed, clear_state_changed
from pybuda import PyBudaModule
from .tensor import Tensor, to_pt_tensors, to_buda_tensors
from . import ci, utils
from pybuda.tools.net2reportify import net2placement

LAST_SUCCESSFUL_STAGE = None
def init_log_last_successful_compile_stage():
    global LAST_SUCCESSFUL_STAGE
    LAST_SUCCESSFUL_STAGE = None


def dump_compiler_cfg(backend_output_directory, compiler_cfg, graph_name):
    import yaml
    try:
        int(os.environ["PYBUDA_DUMP_CONFIG"])
        path = f"{graph_name}_config.yaml"
    except ValueError:
        path = os.environ["PYBUDA_DUMP_CONFIG"]
    with open(os.path.join(backend_output_directory, path), "w") as fd:
        yaml.dump(compiler_cfg.to_dict(), fd, indent=2)


def load_compiler_cfg(compiler_cfg, clobber=False):
    import yaml
    import json
    path = os.environ["PYBUDA_LOAD_CONFIG"]
    loader = json.load if os.path.splitext(path)[1] == ".json" else lambda f: yaml.load(f, yaml.SafeLoader)
    with open(path) as fd:
        d = compiler_cfg.to_dict()
        overrides = loader(fd)
        for k, v in overrides.items():
            d[k] = v
        return CompilerConfig.from_dict(d)


def generate_override_config(graph, balancer_solution, placer_solution, nop_instructions, graph_name):
    import yaml
    try:
        int(os.environ["PYBUDA_GENERATE_OVERRIDE_CONFIG"])
        path = f"{graph_name}_override_config.yaml"
    except ValueError:
        path = os.environ["PYBUDA_GENERATE_OVERRIDE_CONFIG"]

    overrides = {}
    overrides["balancer_op_overrides"] = {k: {
        "grid_shape": [v.grid_shape.r, v.grid_shape.c],
        "t_stream_dir": str(v.t_stream_factor.dir).split(".")[1],
        "t_stream_shape": [v.t_stream_factor.r, v.t_stream_factor.c],
        "fracture_factor": v.fracture_factor,
    } for k, v in balancer_solution.op_models.items()}

    overrides["buffering_nops_to_insert"] = [NopInsertionInstruction.to_json(n) for n in nop_instructions]

    overrides["insert_queues"] = list(list(v) for v in balancer_solution.cut_edges_as_override(graph))

    with open(path, "w") as fd:
        yaml.dump(overrides, fd, indent=2)

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

    pass_specific_output_kwargs: Dict[str, Any] = {}

@dataclass
class CompileContext:
    dev: TTDevice
    graph_name: str
    inputs: Tuple[Union[Tensor, List[Any], Dict[str, Any]],...]
    compiler_cfg: CompilerConfig
    verify_cfg: VerifyConfig
    microbatch_size: int
    microbatch_count: int
    graph: Optional[Graph] = None
    losses: Optional[List[Tensor]] = None
    output_kwargs: Dict[str, Any] = field(default_factory=dict)
    targets: List[Tensor] = field(default_factory=list)
    initial_graph: Optional[Graph] = None
    final_graph: Optional[Graph] = None
    stage: CompileDepth = CompileDepth.INIT_COMPILE
    initial_graph_copy: Optional[Graph] = None
    outputs: Tuple[Tensor, ...] = field(default_factory=tuple)
    intermediate_tensors: Dict[int, Tensor] = field(default_factory=dict)
    parameter_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    input_grads: List[torch.Tensor] = field(default_factory=list)
    netlist_filename: Optional[str] = None
    perf_model_results: Optional[Dict[str, float]] = None
    use_interactive_placer: bool = False
    fracture_chip_id_assignments: Dict[str, int] = field(default_factory=dict)
    buda_targets: List[Tensor] = field(default_factory=list)
    buda_losses: List[Tensor] = field(default_factory=list)
    placer_retry_count: int = 0
    backend_output_directory: str = ""
    in_recompile: bool = False
    recompile_count: int = 0
    target_cycles_offset: int = 0

def calculate_grads(
        outputs: Tuple[Tensor, ...],
        device: "TTDevice",
        intermediate_golden_tensors: Dict,
        is_buda: bool,
        losses=None):
    """
    Verify graph vs. pytorch golden
    """

    # retain intermediate gradients for verification
    for t in intermediate_golden_tensors.values():
        if t.requires_grad == True:
            t.retain_grad()

    # Calculate pytorch gradients
    run_backward = False
    for o in outputs:
        # Check if we need to run, or if gradients have been calculated already
        if o.value().grad is None and o.requires_grad:
            run_backward = True
            break

    if not losses or run_backward:

        if losses is None and device.loss_module is None:
            losses = _generate_random_losses(outputs, is_buda)

        if run_backward:
            _run_pytorch_backward(outputs, device, losses)

    return losses

def pybuda_compile_from_context(context: CompileContext) -> CompileResults:
    """
    Run front-end compile passes and generate a Buda netlist, with a given compile context.

    Parameters
    ----------
    context: CompileContext
        Contains all needed info to run compile passes.

    Returns
    -------
    CompileResults

    """

    # Map stages to functions which execute them.
    stage_to_func = {
        CompileDepth.INIT_COMPILE: init_compile,
        CompileDepth.GENERATE_INITIAL_GRAPH: generate_initial_graph,
        CompileDepth.POST_INITIAL_GRAPH_PASS: run_post_initial_graph_pass,
        CompileDepth.CONSTEVAL_GRAPH: run_consteval_pass,
        CompileDepth.POST_PATTERN_MATCHER: run_post_pattern_matcher,
        CompileDepth.OPTIMIZED_GRAPH: run_optimization_pass,
        CompileDepth.AUTOGRAD: run_autograd_pass,
        CompileDepth.POST_AUTOGRAD_PASS: run_post_autograd_pass,
        CompileDepth.PRE_LOWERING_PASS: run_pre_lowering_pass,
        CompileDepth.FINISH_COMPILE: finish_compile,
    }

    while context.stage != CompileDepth.FULL:
        logger.info("Running compile stage {}", context.stage.name.lower())

        current_stage = context.stage
        verify_cfg = context.verify_cfg
        compiler_cfg = context.compiler_cfg
        dev = context.dev

        # Execute the current stage.
        next_stage = stage_to_func[current_stage](context)

        # Check if we need to stop compilation or perform verifications in the current stage.
        should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, current_stage)

        can_verify = current_stage != CompileDepth.INIT_COMPILE and current_stage != CompileDepth.PRE_LOWERING_PASS and current_stage != CompileDepth.POST_PATTERN_MATCHER
        should_verify = current_stage == CompileDepth.POST_AUTOGRAD_PASS and verify_cfg.verify_post_autograd_passes


        if (verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation) or should_verify) and can_verify:
            in_training = context.compiler_cfg.enable_training and current_stage.value >= CompileDepth.AUTOGRAD.value

            do_verify(current_stage.name.lower(), in_training, context.graph, context.inputs, context.parameter_dict, context.input_grads, context.outputs, dev, context.intermediate_tensors, verify_cfg, False, losses=context.losses, targets=context.targets)

        if should_early_stop_compilation:
            logger.info("Early stopping compilation at stage {}", current_stage.name.lower())
            return generate_compile_results(context.verify_cfg, context.initial_graph_copy, context.outputs, context.intermediate_tensors, context.final_graph, pass_specific_output_kwargs=context.output_kwargs)

        context.stage = next_stage

    return generate_compile_results(context.verify_cfg, context.initial_graph_copy, context.outputs, context.intermediate_tensors, context.final_graph, pass_specific_output_kwargs=context.output_kwargs)

def pybuda_compile(
        dev: TTDevice,
        graph_name: str,
        *inputs: Union[Tensor, List[Any], Dict[str, Any]],
        targets: List[Tensor] = [],
        compiler_cfg: Optional[CompilerConfig] = None,
        verify_cfg: Optional[VerifyConfig] = None,
        losses: Optional[List[Tensor]] = None,
        microbatch_size: int = 1,
        microbatch_count: int = 1) -> CompileResults:
    """
    Run front-end compile passes and generate a Buda netlist for given input tensors. Optionally verify
    against PyTorch model.

    This version has significant amount of verification built-in, and is primarily used for testing. A "deliverable"
    version that does only the compile will be written in the future.

    Parameters
    ----------
    dev: TTDevice
        Device to compile modules for. Modules should already be placed on the device.

    graph_name: str
        Name to be used in the netlist

    *inputs: Tuple[Tensor, ...]
        Input tensors to compile for. Tensors must have set shapes, but values are only needed for
        automatic verification.

    targets: List[Tensor], optional
        Optional list of target tensors, if this device has a loss module

    verify_cfg: Optional[VerifyConfig]
        If set, automatic verification vs. pytorch golden result will be performed, with given parameters
        must contain data.


    Returns
    -------
    CompileResults

    """

    inputs = list(inputs)
    if verify_cfg is None:
        verify_cfg = VerifyConfig.disabled() # no verification config provided, disable by default

    if compiler_cfg is None:
        compiler_cfg = _get_global_compiler_config()

    compiler_cfg.apply_env_config_overrides()

    compile_context: CompileContext = CompileContext(
        dev=dev,
        graph_name=graph_name,
        inputs=inputs,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
        microbatch_size=microbatch_size,
        microbatch_count=microbatch_count,
        targets=targets,
        losses=losses,
    )

    return pybuda_compile_from_context(compile_context)

def check_for_compilation_early_stop(desired_stage, current_stage):
    """
    Determines should current compilation process stop or not based on desired
    and current phase of execution.

    Parameters
    ----------
    desired_stage: CompileDepth
        Desired phase for compiler early stopping.

    current_stage: CompileDepth
        Current phase for compiler early stopping.

    Returns
    -------
    Boolean
    """
    # update global compile stage variable
    global LAST_SUCCESSFUL_STAGE
    LAST_SUCCESSFUL_STAGE = str(CompileDepth(current_stage.value).name)

    if not CompileDepth.has_value(desired_stage.value):
        raise Exception("Invalid compilation depth flag: {}".format(desired_stage.name))

    if desired_stage.value == current_stage.value:
        logger.info("Compilation early stopping after {}".format(current_stage.name))

        return True

    return False

def placer_breaks_eval(value):
    if type(value) is query.NodePredicateBuilder:
        return value.eval()
    elif type(value) is list:
        return [placer_breaks_eval(v) for v in value]
    else:
        assert type(value) is str
        return value

def placer_op_overrides_eval(value):
    assert type(value) is tuple
    if type(value[0]) is query.NodePredicateBuilder:
        return (value[0].eval(), value[1])
    else:
        return value


def generate_compile_results(
    verify_cfg = None,
    initial_graph = None,
    outputs = None,
    intermediate_tensors = None,
    final_graph = None,
    *,
    pass_specific_output_kwargs = None,
):
    """
    Wrapper for generating result from the graph compiler. Contains initial and final graphs, output tensors,
    and, optionally golden results for final output and intermediates, if desired.

    Parameters
    ----------
    verify_cfg: VerifyConfig
        Value verification config

    initial_graph: Graph
        Initial graph, immediately after conversion from the input framework

    outputs: Tuple[Tensor, ...]
        Output tensors

    intermediate_tensors: Dict[str, Tensor]
        Intermediated tensors

    final_graph: Graph
        Buda graph

    netlist_filename: str
        Netlist file name

    Returns
    -------
    CompileResults
    """
    ret = CompileResults()

    ret.initial_graph = initial_graph
    ret.outputs = outputs
    if verify_cfg and verify_cfg.intermediates:
        ret.golden_intermediates = {
            initial_graph.get_node_name(node_id): tensor
            for node_id, tensor in intermediate_tensors.items() if initial_graph.has_node_with_id(node_id)
        }
    ret.final_graph = final_graph

    if outputs:
        ret.golden_outputs = [out.value() if out.has_value() else None for out in outputs]

    if pass_specific_output_kwargs:
        ret.pass_specific_output_kwargs = pass_specific_output_kwargs

    return ret

def init_compile(context: CompileContext) -> CompileDepth:
    
    dev = context.dev
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    targets = context.targets

    force_full = bool(int(os.environ.get("PYBUDA_FORCE_FULL_COMPILE_DEPTH", "0")))
    if force_full:
        compiler_cfg.compile_depth = CompileDepth.FULL

    if len(targets) > 0:
        assert dev.loss_module is not None, "Target provided for compilation, but this device has no loss module"

    if dev.loss_module is not None:
        assert len(targets) > 0, f"Device {dev} has a loss module, but no targets were provided for compilation"

    context.backend_output_directory = compiler_cfg.backend_output_dir
    ci.initialize_output_build_directory(context.backend_output_directory)

    # compiler_cfg is fully formed
    if "PYBUDA_LOAD_CONFIG" in os.environ:
        compiler_cfg = load_compiler_cfg(compiler_cfg)
    elif "PYBUDA_DUMP_CONFIG" in os.environ:
        dump_compiler_cfg(context.backend_output_directory, compiler_cfg, graph_name)

    init_log_last_successful_compile_stage()

    return CompileDepth.GENERATE_INITIAL_GRAPH

def generate_initial_graph(context: CompileContext) -> CompileDepth:
    """
    Generates initial graph from the input framework.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """

    if context.compiler_cfg.compile_tvm_to_python and context.dev.graph is None:
        module_inputs = context.inputs
        for index, module in enumerate(context.dev.modules):
            if not isinstance(module, PyBudaModule):
                from .tvm_to_python import generate_pybuda_module
                prev_state = state_changed()
                modules, dev_types, module_inputs = generate_pybuda_module(module, to_pt_tensors(module_inputs), context.compiler_cfg, module.name, context.verify_cfg,)
                assert len(modules) == 1, "Attemping to load split model onto single devices"
                context.dev.modules[index] = modules[0]

                if index == 0:
                    context.inputs = module_inputs

                if not(prev_state):
                    clear_state_changed()

            if index < len(context.dev.modules) - 1 and not context.compiler_cfg.compile_subgraphs:
                if module is context.dev.loss_module:
                    if len(module_inputs) == 1:
                        module_inputs = context.dev.modules[index].forward(module_inputs[0], context.targets[0])
                    else:
                        module_inputs = context.dev.modules[index].forward(tuple(module_inputs), tuple(context.targets))
                else:
                    module_inputs = context.dev.modules[index].forward(*module_inputs)

                if isinstance(module_inputs, Tensor):
                    module_inputs = (module_inputs,) # Force a tuple

    if context.dev.graph is None:
        context.graph, context.outputs, context.intermediate_tensors, context.inputs, _ = context.dev.generate_graph(*context.inputs, return_intermediate=context.verify_cfg.intermediates, graph_name=context.graph_name, compiler_cfg=context.compiler_cfg, target_tensors=context.targets, verify_cfg=context.verify_cfg)
    else:
        context.graph = context.dev.graph
        context.intermediate_tensors = context.dev.intermediate_tensors
        context.outputs = context.dev.output_tensors

    context.graph.set_microbatch(context.microbatch_size)
    dump_graph(context.graph, context.graph_name, "initial_graph")
    if context.compiler_cfg.enable_link_past_cache_ios:
        # move index ops to weights if applicable
        move_index_to_mm_weights(context.graph)

        # link past cache ios will change the number on inputs / outputs, so it is called before we clone the initial graph
        new_params = link_past_cache_ios(context.graph)
        inputs_to_remove = []
        for k, v in new_params.items():
            context.dev.modules[-1].add_parameter(k, Parameter(context.inputs[v].value(), requires_grad=False, name=k))
            inputs_to_remove.append(context.inputs[v])
        for i in inputs_to_remove:
            context.inputs.remove(i)

    context.initial_graph_copy = context.graph.clone() # save the original graph for verification and analysis
    context.input_grads = []
    context.parameter_dict = {p.get_name() : p.value(is_buda=False) for p in context.dev.get_parameters()}

    return CompileDepth.POST_INITIAL_GRAPH_PASS

def run_post_initial_graph_pass(context: CompileContext) -> CompileDepth:
    """
    Runs post initial graph passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    graph, intermediate_tensors = context.graph, context.intermediate_tensors

    if compiler_cfg.enable_consteval:
        run_consteval_graph_pass(graph)
    inserted_node_id_mapping, context.fracture_chip_id_assignments = run_post_initial_graph_passes(graph, compiler_cfg, compiler_cfg.fracture_groups)

    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        # If we have multi-level of decomposition, some node id might not in the original
        # intermediate tensor dict.
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "decomposed_graph")

    next_stage = CompileDepth.OPTIMIZED_GRAPH
    if compiler_cfg.enable_consteval:
        next_stage = CompileDepth.CONSTEVAL_GRAPH
    elif compiler_cfg.match_subgraph_patterns:
        next_stage = CompileDepth.POST_PATTERN_MATCHER

    return next_stage

def run_consteval_pass(context: CompileContext) -> CompileDepth:
    """
    Runs consteval pass.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph = context.graph
    graph_name = context.graph_name

    run_consteval_graph_pass(graph)
    dump_graph(graph, graph_name, "consteval_graph")

    next_stage = CompileDepth.OPTIMIZED_GRAPH
    if compiler_cfg.match_subgraph_patterns:
        next_stage = CompileDepth.POST_PATTERN_MATCHER

    return next_stage

def run_post_pattern_matcher(context: CompileContext) -> CompileDepth:
    """
    Runs post pattern matcher passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph = context.graph
    graph_name = context.graph_name

    graph, match_result = pypattern_matcher.lower_pybuda_to_pattern_matcher(graph, compiler_cfg.match_subgraph_patterns)
    context.output_kwargs["match_result"] = match_result

    if match_result.is_subgraph_loopable:
        dump_graph(graph, graph_name, "looped_graph")

    return CompileDepth.OPTIMIZED_GRAPH

def run_optimization_pass(context: CompileContext) -> CompileDepth:
    """
    Runs optimization passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph, intermediate_tensors = context.graph, context.intermediate_tensors

    run_optimization_graph_passes(graph)
    dump_graph(graph, graph_name, "optimized_graph")

    inserted_node_id_mapping = run_post_optimize_decompose_graph_passes(graph, compiler_cfg)
    dump_graph(graph, graph_name, "decomposed_optimized_graph")
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    # Workaround for TVM and lack of parameters at the time optimizer is created
    if dev.optimizer:
        if dev.optimizer.device_params:
            dev.optimizer.set_parameters_to_optimize(dev.modules[0].get_parameters())
        dev.optimizer.set_optimizer_parameters()

    next_stage = CompileDepth.POST_AUTOGRAD_PASS
    if context.compiler_cfg.enable_training:
        next_stage = CompileDepth.AUTOGRAD

    return next_stage

def run_autograd_pass(context: CompileContext) -> CompileDepth:
    """
    Runs autograd pass.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph, intermediate_tensors, outputs = context.graph, context.intermediate_tensors, context.outputs

    autograd_config = pyautograd.AutogradConfig(recompute=compiler_cfg.enable_recompute, optimizer=dev.optimizer)
    autograd_engine = pyautograd.AutogradEngine(graph, autograd_config)

    graph = autograd_engine.run()
    dump_graph(graph, graph_name, "post_autograd")

    context.losses = calculate_grads(outputs, dev, intermediate_tensors, False, context.losses)

    # Record calculated input grads from the previous do_verify call and save so that we don't keep re-calculating and
    # accumulating on each verification call
    context.input_grads = [i.value().grad for i in context.inputs if i.value().requires_grad and i.value().grad is not None]

    return CompileDepth.POST_AUTOGRAD_PASS

def run_post_autograd_pass(context: CompileContext) -> CompileDepth:
    """
    Runs post autograd passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph, intermediate_tensors, losses, outputs = context.graph, context.intermediate_tensors, context.losses, context.outputs

    inserted_node_id_mapping = run_post_autograd_graph_passes(graph, compiler_cfg)
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "post_autograd_passes")
    if compiler_cfg.enable_training:
        calculate_grads(outputs, dev, intermediate_tensors, False, losses)

    return CompileDepth.PRE_LOWERING_PASS

def run_pre_lowering_pass(context: CompileContext) -> CompileDepth:
    """
    Runs pre lowering passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph = context.graph

    run_lower_to_mlir_passes(graph)
    dump_graph(graph, graph_name, "pre_lowering")

    for parameter in dev.get_parameters():
        parameter._set_fp32_fallback(dev.fp32_fallback)

    context.final_graph = graph
    return CompileDepth.FINISH_COMPILE


def finish_compile(context: CompileContext) -> CompileDepth:
    """
    Runs backend golden verify.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    verify_cfg = context.verify_cfg
    compiler_cfg = context.compiler_cfg
    dev = context.dev

    context.output_kwargs["consteval_trace"] = pygraph.record_consteval_operations(context.final_graph)
    compile_results = generate_compile_results(
        verify_cfg,
        context.initial_graph_copy, context.outputs,
        context.intermediate_tensors,
        pass_specific_output_kwargs = context.output_kwargs
    )

    return CompileDepth.FULL
