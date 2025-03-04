# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from forge.tensor import to_tf_tensors, to_pt_tensors
from forge.tvm_utils import flatten_inputs, flatten_structured_output
from forge.execution_tracker import ExecutionStage, record_execution_phase_and_stage
import torch

import numpy as np
import tvm
from tvm.ir.transform import Pass
import tvm.relay as relay
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
    _construct_concrete_function,
)
from loguru import logger

from ctypes import c_void_p
import copy
import json

from collections import OrderedDict
from collections.abc import MutableMapping

from json import JSONEncoder
import os
import subprocess
from os.path import exists as file_exists
import onnxruntime as ort
import onnx
import onnx.numpy_helper
import mxnet as mx
from tvm.relay.expr import Tuple
from forge.tvm_calls.relay.op.forge import verify_tvm_compile, flatten_IO, compile_for_forge, partition_for_forge
from jax.experimental import jax2tf
from jax.tools.jax_to_ir import tf_wrap_with_input_names
import collections
from transformers.utils.generic import ModelOutput
from forge.tvm_calls.forge_utils import (
    extract_framework_model_outputs,
    extract_flatten_inputs,
    construct_tvm_ir,
    extract_function_callnodes,
    trace_to_origin,
    has_op,
)
import hashlib

dev_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "tt"}
cpu_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "cpu"}


def retrieve_graph(json_graph, t):
    function_name = t[0]
    if function_name in json_graph["functions"]:
        return

    json_graph["functions"][function_name] = t[1]
    json_graph["param_names"][function_name] = t[2]


@tvm.register_func
def retrieve_forge_json_graph(*args):
    t = tuple(args)
    global dev_json_graph
    retrieve_graph(dev_json_graph, t)


@tvm.register_func
def retrieve_forge_cpudevice_json_graph(*args):
    t = tuple(args)
    global cpu_json_graph
    retrieve_graph(cpu_json_graph, t)


def load_tvm_graph(inputs, module, compiler_cfg, graph_name, framework, path=None, verify_cfg=None, input_names=[]):
    """
    Loads TVM graph ported to the Forge from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process from specific framework to the Forge graph representation, or
    skip this compilation process and load serialize TVM graph which is already ported to Forge
    by initial invocation.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations

    path: str
        Path to onnx file on disk. This is used to verify TVM results vs. framework results.

    Returns
    -------
    Dictionary, OrderedDict, Tuple, Boolean
        TVM ported graph, Weights, Input tensors, Constant evaluation
    """
    if compiler_cfg.tvm_graph_store_path != "" and compiler_cfg.tvm_graph_load_path != "":
        logger.warning(f"TVM serialization logic will be skipped as both store and load paths are provided")

    json_graphs, flattened_inputs = compile_tvm_graph(
        inputs,
        module,
        compiler_cfg,
        graph_name=graph_name,
        input_names=input_names,
        path=path,
        verify_cfg=verify_cfg,
        framework=framework,
    )

    flattened_pytorch_inputs, weights = format_tvm_graph_weights(
        flattened_inputs, module, compiler_cfg, framework=framework
    )

    serialize_and_store_tvm_graph(json_graphs, compiler_cfg, framework=framework)

    return json_graphs, flattened_pytorch_inputs, weights


def compile_tvm_graph(
    inputs, module, compiler_cfg, graph_name, input_names=[], path=None, verify_cfg=None, framework=None
):
    """
    Compiles TVM graph ported to the Forge from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process or only load serilized TVM graph and thus increase test performance.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device

    compiler_cfg: CompilerConfig
        Compiler configurations

    path: str
        Path to onnx file on disk. This is used to verify TVM results vs. framework results.

    Returns
    -------
    Dictionary
        TVM ported graph
    """
    # clear out graphs before starting in case python session is ongoing
    global dev_json_graph
    global cpu_json_graph
    dev_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "tt"}
    cpu_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "cpu"}

    if framework == "pytorch":
        json_graphs, inputs = compile_pytorch_for_forge(
            module,
            *inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
            input_names=input_names,
        )
    elif framework == "tensorflow":
        # convert pytorch tensors to tf tensors
        tf_inputs = to_tf_tensors(inputs)
        json_graphs, inputs = compile_tf_for_forge(
            module, *tf_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg
        )
    elif framework == "tf_graphdef":
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            tf_inputs = tuple(None if t is None else tf.convert_to_tensor(t.detach().numpy()) for t in inputs)
        else:
            tf_inputs = inputs
        json_graphs = compile_tf_graphdef_for_forge(
            module,
            *tf_inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
        )
    elif framework == "onnx":
        assert all([isinstance(x, torch.Tensor) for x in inputs])
        onnx_inputs = [x.detach().numpy() for x in inputs]
        json_graphs, _ = compile_onnx_for_forge(
            module, path, *onnx_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg
        )
    elif framework == "mxnet":
        assert all([isinstance(x, torch.Tensor) for x in inputs])
        mxnet_inputs = [mx.nd.array(x.detach().numpy()) for x in inputs]
        json_graphs = compile_mxnet_for_forge(
            module, *mxnet_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg
        )
    elif framework == "jax":
        tf_inputs = to_tf_tensors(inputs, force_float32=True)
        json_graphs, inputs = compile_jax_for_forge(
            module, *tf_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg
        )
    elif framework == "tflite":
        tf_inputs = to_tf_tensors(inputs, force_float32=True)
        json_graphs, inputs = compile_tflite_for_forge(
            module, path, *tf_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg
        )
    else:
        raise RuntimeError(f"Unsupported module type {type(module)}")

    return json_graphs, inputs


def save_nid_to_input_idx(traced_model_inputs, json_graph):
    existing_graph = json.loads(json_graph["graph"])

    nid_to_input_idx = {}

    # reorder arg nodes to be inputs first, then parameters
    for i, arg_idx in enumerate(existing_graph["arg_nodes"]):
        name = existing_graph["nodes"][arg_idx]["name"]
        if name not in traced_model_inputs:
            continue

        nid_to_input_idx[arg_idx] = traced_model_inputs.index(name)

    json_graph["nid_to_input_idx"] = nid_to_input_idx


def extract_graphs(partitioned_mod, forge_params, input_names, weight_names, param_name_lookup={}, graph_hash=""):
    mod = partitioned_mod["main"]
    main_graph = str(mod.astext())
    cpu_json_graph["hash"] = graph_hash
    dev_json_graph["hash"] = graph_hash

    assert (
        len(cpu_json_graph["functions"].keys()) <= 2
    ), "At most two cpu functions should exist (one pre and one post). If there are more they should have been merged into one during tvm partitioning."
    cpu_functions = list(cpu_json_graph["functions"].keys())
    assert (
        len(dev_json_graph["functions"].keys()) <= 1
    ), "At most one device function should exist. If there are more they should have been merged into one during tvm partitioning."
    device_function = list(dev_json_graph["functions"].keys())[0]

    cpu_pre_function = None
    cpu_post_function = None

    func_callnodes = extract_function_callnodes(partitioned_mod["main"], partitioned_mod.get_global_vars())
    for node in func_callnodes:
        if node.op.name_hint == device_function:
            for arg in node.args:
                op = None
                if isinstance(arg, tvm.relay.expr.TupleGetItem):
                    op = arg.tuple_value.op
                elif isinstance(arg, tvm.relay.expr.Call):
                    op = arg.op
                if op is not None:
                    if op.name_hint != cpu_pre_function:
                        if not cpu_pre_function:
                            cpu_pre_function = op.name_hint
                        else:
                            assert (
                                op.name_hint == cpu_pre_function
                            ), "There is more than one cpu pre function. This should not be possible. They should have been merged into one during tvm partitioning."
    if cpu_pre_function is not None:
        cpu_functions.remove(cpu_pre_function)
    assert (
        len(cpu_functions) <= 1
    ), "There is more than one cpu post function. This should not be possible. They should have been merged into one during tvm partitioning."
    cpu_post_function = cpu_functions[0] if len(cpu_functions) else None

    if cpu_pre_function is not None:
        cpu_pre_json_graph = copy.deepcopy(cpu_json_graph)
        cpu_pre_json_graph["graph"] = cpu_json_graph["functions"][cpu_pre_function]

        # Only keep the pre function in the pre json
        functions_to_remove = []
        for function in cpu_pre_json_graph["functions"]:
            if function != cpu_pre_function:
                functions_to_remove.append(function)

        for func in functions_to_remove:
            del cpu_pre_json_graph["functions"][func]

        cpu_pre_json_graph["params"] = {}
        for function_name in forge_params.keys():
            if function_name == cpu_pre_function:
                cpu_pre_json_graph["params"].update(
                    {
                        name: v.numpy()
                        for (k, v), name in zip(
                            forge_params[function_name].items(), cpu_pre_json_graph["param_names"][function_name]
                        )
                    }
                )
    else:
        cpu_pre_json_graph = {"graph": ""}

    dev_json_graph["graph"] = dev_json_graph["functions"][device_function]

    dev_json_graph["params"] = {}
    for function_name in forge_params.keys():
        if function_name in dev_json_graph["param_names"]:
            dev_json_graph["params"].update(
                {
                    name: v.numpy()
                    for (k, v), name in zip(
                        forge_params[function_name].items(), dev_json_graph["param_names"][function_name]
                    )
                }
            )

    if cpu_post_function is not None:
        cpu_post_json_graph = copy.deepcopy(cpu_json_graph)
        cpu_post_json_graph["graph"] = cpu_json_graph["functions"][cpu_post_function]

        # Only keep the post function in the post json
        functions_to_remove = []
        for function in cpu_post_json_graph["functions"]:
            if function != cpu_post_function:
                functions_to_remove.append(function)

        for func in functions_to_remove:
            del cpu_post_json_graph["functions"][func]

        cpu_post_json_graph["params"] = {}
        for function_name in forge_params.keys():
            if function_name == cpu_post_function:
                cpu_post_json_graph["params"].update(
                    {
                        name: v.numpy()
                        for (k, v), name in zip(
                            forge_params[function_name].items(), cpu_post_json_graph["param_names"][function_name]
                        )
                    }
                )
    else:
        cpu_post_json_graph = {"graph": ""}

    json_graphs = []
    if cpu_pre_function is not None:
        save_nid_to_input_idx(input_names, cpu_pre_json_graph)  # Input order might not be preserved by TVM
        cpu_pre_json_graph["num_forge_inputs"] = len(input_names)
        json_graphs.append(
            copy.deepcopy(
                clean_names(
                    json_graph=cpu_pre_json_graph, forge_params=forge_params, param_name_lookup=param_name_lookup
                )
            )
        )
    else:
        save_nid_to_input_idx(input_names, dev_json_graph)  # Input order might not be preserved by TVM
        dev_json_graph["num_forge_inputs"] = len(input_names)

    json_graphs.append(
        copy.deepcopy(
            clean_names(json_graph=dev_json_graph, forge_params=forge_params, param_name_lookup=param_name_lookup)
        )
    )

    if cpu_post_json_graph["graph"] != "":
        json_graphs.append(
            copy.deepcopy(
                clean_names(
                    json_graph=cpu_post_json_graph, forge_params=forge_params, param_name_lookup=param_name_lookup
                )
            )
        )

    return json_graphs


class ConvertEmulatedDtypes:
    """
    This class converts data formats which must be emulated by CPU into float32 within its context.
    It is used when a model's parameters and inputs would be slow to compute on CPU and precision is not important.
    """

    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs
        self.emulated_dfs = [torch.bfloat16]
        self.fallback = torch.float32

    def flatten_object(self, obj):
        if isinstance(obj, (list, tuple)):
            return [item for sublist in obj for item in self.flatten_object(sublist)]
        elif isinstance(obj, dict):
            return [item for key, value in obj.items() for item in self.flatten_object(value)]
        else:
            return [obj]

    def __enter__(self):
        # Convert emulated model parameters to fallback
        self.param_dfs = {}
        for name, param in self.model.named_parameters():
            if param.dtype in self.emulated_dfs:
                self.param_dfs[name] = param.dtype
                param.data = param.data.to(self.fallback)

        # Convert emulated inputs to fallback
        self.input_dfs = []
        for inp in self.flatten_object(self.inputs):
            self.input_dfs.append(inp.dtype)
            if inp.dtype in self.emulated_dfs:
                inp.data = inp.data.to(self.fallback)

    def __exit__(self, *args):
        # Convert model parameters back to original dtype
        for name, param in self.model.named_parameters():
            if name in self.param_dfs:

                param.data = param.data.to(self.param_dfs[name])

        # Convert inputs back to original dtype
        for inp, df in zip(self.flatten_object(self.inputs), self.input_dfs):
            inp.data = inp.data.to(df)


def compile_pytorch_for_forge(torchmod, *inputs, graph_name, compiler_cfg, verify_cfg=None, input_names=[]):
    training_mode = torchmod.training

    with ConvertEmulatedDtypes(torchmod, inputs):
        # Extract framework model outputs
        framework_outputs = extract_framework_model_outputs(
            framework="pytorch",
            model=torchmod,
            inputs=inputs,
            verify_tvm_compile=verify_cfg.verify_tvm_compile,
        )

        # (Temporary): Remove when forge supports dropout
        if training_mode and compiler_cfg.enable_tvm_dropout == False:
            torchmod.eval()

        if isinstance(torchmod, torch.jit.ScriptModule):
            torchmod.eval()
            torchmod = torch.jit.freeze(torchmod)

        # Trace framework model
        traced_model = torch.jit.trace(torchmod, inputs, strict=False)

    # Extract flatten inputs
    flattened_inputs, flattened_input_names, flattened_name_map, input_structure = extract_flatten_inputs(
        framework="pytorch",
        model=traced_model,
        inputs=inputs,
        input_names=input_names,
    )

    graph_string = traced_model.graph.str().encode("utf-8")
    m = hashlib.sha256()
    m.update(graph_string)
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, m.hexdigest(), framework="pytorch")
    if cached_graphs is not None:
        return cached_graphs, flattened_inputs

    # Generate TVM module
    convert_params = compiler_cfg.convert_framework_params_to_tvm
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_structure, do_convert_params=convert_params)
    record_execution_phase_and_stage(ExecutionStage.TVM_GENERATE_RELAY_IRMODULE)
    logger.trace("From PyTorch")
    logger.trace(mod.functions)

    mod = flatten_IO(mod, flattened_name_map)
    record_execution_phase_and_stage(ExecutionStage.TVM_FLATTEN_IO)

    # Construct TVM IR
    mod, _ = construct_tvm_ir(
        framework="pytorch",
        model=torchmod,
        tvm_mod=mod,
        params=params,
        compiler_cfg=compiler_cfg,
    )

    # Construct NumPy inputs
    flattened_inputs_as_float = (act.float() if torch.is_floating_point(act) else act for act in flattened_inputs)
    np_inputs = {name: inp.detach().numpy() for name, inp in zip(flattened_input_names, flattened_inputs_as_float)}

    # Compile TVM for Forge
    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        np_inputs,
        framework_outputs,
        input_names=flattened_input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    if training_mode:
        torchmod.train()

    # Extract Graphs (TT, CPU, ...)
    json_graphs = extract_graphs(
        partitioned_mod, forge_params, flattened_input_names, torchmod.state_dict().keys(), graph_hash=m.hexdigest()
    )

    return json_graphs, flattened_inputs


def compile_tvm_for_forge(
    mod,
    params,
    inputs,
    golden_outputs,
    graph_name,
    input_names=[],
    return_params=False,
    compiler_cfg=None,
    verify_cfg=None,
):
    target = "llvm"
    verify_args = {"inputs": inputs, "framework_outputs": golden_outputs, "verify_cfg": verify_cfg}
    mod, params = compile_for_forge(mod, target=target, params=params, graph_name=graph_name, **verify_args)

    if verify_cfg is not None and verify_cfg.verify_tvm_compile:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot verify TVM compile without converting framework params to relay"
        # If we have conv2d_transpose ops that are channel-last, tvm cannot execute the module, skip in this case
        skip_verify = has_op(mod["main"], "nn.conv2d_transpose", {"data_layout": "NHWC"})
        if skip_verify:
            logger.warning(
                "Module contains a channel-last nn.conv2d_transpose op, this is not supported in TVM (but may be supported in Forge). Skipping verification..."
            )
        else:
            verify_tvm_compile(mod, params, inputs, target, golden_outputs, "compile_for_forge", verify_cfg=verify_cfg)

    # Reconstruct Ops + export forge graph
    mod, forge_params = partition_for_forge(
        mod, graph_name=graph_name, compiler_cfg=compiler_cfg, input_names=input_names
    )
    tvm.relay.build_module.build(mod, target=target, params=params)
    record_execution_phase_and_stage(ExecutionStage.TVM_GRAPH_PARTITIONING)

    if return_params:
        return mod, forge_params
    else:
        return mod


def clean_names(json_graph, forge_params, param_name_lookup={}):
    precursor = (
        "tvmgen_default_forge_main_" if json_graph["device"] != "cpu" else "tvmgen_default_forge_cpudevice_main_"
    )

    def trim_count(name):
        for i in range(-1, -len(name), -1):
            if name[i] == "_":
                return name[:i], name[i + 1 :]

    if len(json_graph["params"]) > 0:

        old_params = json_graph["params"]
        json_graph["params"] = {}
        for k, v in old_params.items():
            if precursor in k:
                num_digits = k.replace(precursor, "").find("_")
                key = k.replace(precursor, "")[num_digits + 1 :] + k.replace(precursor, "")[:num_digits]
            else:
                name, num_digits = trim_count(k)
                old_name = param_name_lookup[name] if name in param_name_lookup else name
                param_name_lookup[k] = old_name + f"_{num_digits}"
                key = param_name_lookup[k]  # This is done to sync the node names with the param names
            json_graph["params"][key] = v

    graph = json.loads(json_graph["graph"])

    for node in graph["nodes"]:
        if precursor in node["name"]:
            num_digits = node["name"].replace(precursor, "").find("_")
            node["name"] = (
                node["name"].replace(precursor, "")[num_digits + 1 :] + node["name"].replace(precursor, "")[:num_digits]
            )
        elif param_name_lookup is not None and node["name"] in param_name_lookup:
            node["name"] = param_name_lookup[node["name"]]

    json_graph["graph"] = json.dumps(graph)

    return json_graph


def duplicate_dequantize_nodes_in_onnx_graph(onnx_module):
    from collections import defaultdict

    # Create a dictionary to store the consumers of each tensor
    tensor_consumers = defaultdict(list)

    graph = onnx_module.graph
    # Populate the tensor_consumers dictionary
    for node in graph.node:
        for input_name in node.input:
            tensor_consumers[input_name].append(node.name)

    # Find and duplicate nodes with output branches
    nodes_to_add = []
    nodes_to_remove = []
    indices_for_adding = []
    for node_ind, node in enumerate(graph.node):

        if node.op_type != "DequantizeLinear":
            continue

        output_name = node.output[0]
        consumers = tensor_consumers[output_name]

        if len(consumers) > 1:
            # Duplicate the node for each consumer
            for i, consumer_name in enumerate(consumers):
                new_node_name = node.name + f"_clone{i}"
                new_output_name = output_name + f"_clone{i}"
                attrs = {"axis": node.attribute[0].i} if len(node.attribute) > 0 else {}
                cloned_node = onnx.helper.make_node(
                    node.op_type, node.input, [new_output_name], name=new_node_name, **attrs
                )

                # Add the cloned node to the list of nodes to add
                nodes_to_add.append(cloned_node)
                indices_for_adding.append((cloned_node, node_ind))

                # Update the consumer to use the cloned node's output
                consumer_node = next(n for n in graph.node if n.name == consumer_name)
                for j, input_name in enumerate(consumer_node.input):
                    if input_name == output_name:
                        consumer_node.input[j] = new_output_name

            # Remove the original node since it will be replaced by its clones
            nodes_to_remove.append(node)

    # This is needed to remain the order of the nodes in graph
    # since graph is not put in topsort order when visiting nodes
    for i, (node, insertion_index) in enumerate(indices_for_adding):
        graph.node.insert(insertion_index + i, node)

    for node in nodes_to_remove:
        graph.node.remove(node)


def compile_onnx_for_forge(onnx_mod, path, *inputs, graph_name, compiler_cfg, verify_cfg=None):
    import onnxruntime as ort

    assert path != None, "Onnx compile needs path to onnx file on disk."

    # Set default num threads to 2, hangs on some hosts otherwise https://github.com/microsoft/onnxruntime/issues/10166
    so = ort.SessionOptions()
    so.inter_op_num_threads = 2
    so.intra_op_num_threads = 2

    ort_sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

    input_names = []
    for inp in ort_sess.get_inputs():
        input_names.append(inp.name)

    input_dict = {}
    input_shape_dict = {}
    for name, tensor in zip(input_names, inputs):
        input_dict[name] = tensor
        input_shape_dict[name] = tensor.shape

    assert len(input_names) == len(inputs), "Number of input names must match number of inputs"

    duplicate_dequantize_nodes_in_onnx_graph(onnx_mod)
    framework_outputs = extract_framework_model_outputs(
        framework="onnx",
        model=onnx_mod,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
        path=path,
        input_dict=input_dict,
    )

    graph_string = str(onnx_mod).encode("utf-8")
    m = hashlib.sha256()
    m.update(graph_string)
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, m.hexdigest(), framework="onnx")
    if cached_graphs is not None:
        return cached_graphs, inputs

    mod, params = relay.frontend.from_onnx(onnx_mod, input_shape_dict, freeze_params=False)
    mod = relay.transform.DynamicToStatic()(mod)
    record_execution_phase_and_stage(ExecutionStage.TVM_GENERATE_RELAY_IRMODULE)

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot use constant prop without converting framework params to relay"
        propped_params = {k: (v, True) for k, v in params.items()}
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        input_dict,
        framework_outputs,
        input_names=input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    weight_names = [weight.name for weight in onnx_mod.graph.initializer]
    json_graphs = extract_graphs(partitioned_mod, forge_params, input_names, weight_names, graph_hash=m.hexdigest())

    return json_graphs, inputs


def compile_tflite_for_forge(module, path, *inputs, graph_name, compiler_cfg, verify_cfg=None):

    assert path != None, "TFLite compile needs path to .tflite file on disk."
    tflite_model_buf = open(path, "rb").read()

    input_details = module.get_input_details()

    framework_outputs = extract_framework_model_outputs(
        framework="tflite",
        model=module,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
        path=path,
    )

    # Get TFLite model from buffer
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    input_shape_dict = {}
    input_dict = {}
    input_names = []

    for i, details in enumerate(input_details):
        input_names.append(details["name"])
        input_shape_dict[details["name"]] = list(details["shape"])
        input_dict[details["name"]] = inputs[i]

    graph_string = str(module).encode("utf-8")
    m = hashlib.sha256()
    m.update(graph_string)
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, m.hexdigest(), framework="tflite")
    if cached_graphs is not None:
        return cached_graphs, inputs

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict=input_shape_dict,
    )
    record_execution_phase_and_stage(ExecutionStage.TVM_GENERATE_RELAY_IRMODULE)

    assert len(input_names) == len(inputs), "Number of input names must match number of inputs"

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot use constant prop without converting framework params to relay"
        propped_params = {k: (v, True) for k, v in params.items()}
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        input_dict,
        framework_outputs,
        input_names=input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    json_graphs = extract_graphs(partitioned_mod, forge_params, input_names, [], graph_hash=m.hexdigest())

    return json_graphs, inputs


from tf2onnx.tf_loader import convert_variables_to_constants_large_model


def get_frozen_graph_for_large_jax_model(
    jaxmodel,
    compiler_cfg,
    *inputs,
):
    # This conversion path will perform the following steps:
    #     - Feed jax parameters as input to jax model
    #     - convert jax model to TF function without parameters
    #     - convert jax parameters to TF Variables
    #     - trace TF function with Variables as input
    #     - use helper function to handle variable freezing (This will avoid using Protobuf)
    #     - attach output shape for each node in frozen graph.

    if "params" in jaxmodel.variables.keys():
        params = jaxmodel.variables["params"]
    else:
        params = {}

    if len(params) == 0:
        tf_fn = jax2tf.convert(jaxmodel, enable_xla=compiler_cfg.enable_xla_jax_convert)
        tf_func = tf.function(tf_fn, autograph=False, jit_compile=True)
        tf_func = tf_func.get_concrete_function(*inputs)
    else:
        predict_fn = lambda params, input: jaxmodel.apply({"params": params}, *input)
        tf_fn = jax2tf.convert(predict_fn, enable_xla=compiler_cfg.enable_xla_jax_convert)

        params_vars = tf.nest.map_structure(lambda param: tf.Variable(param, trainable=True), params)
        tf_func = tf.function(lambda inputs: tf_fn(params_vars, inputs), autograph=False, jit_compile=True)
        # Get graph definition
        tf_func = tf_func.get_concrete_function(inputs)

    graph_def = convert_variables_to_constants_large_model(tf_func)

    # Add Shapes to frozen graph
    for node in graph_def.node:
        op = tf_func.graph._nodes_by_name[node.name]
        if op.outputs:
            node.attr["_output_shapes"].list.shape.extend([output.get_shape().as_proto() for output in op.outputs])

    return graph_def, tf_func


def compile_jax_for_forge(jaxmodel, *inputs, graph_name, compiler_cfg, verify_cfg=None):
    # Extract framework model outputs
    framework_outputs = extract_framework_model_outputs(
        framework="jax",
        model=jaxmodel,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
    )

    if compiler_cfg.enable_tvm_jax_freeze_large_model:
        graph_def, tf_func = get_frozen_graph_for_large_jax_model(
            jaxmodel,
            compiler_cfg,
            *inputs,
        )
    else:
        # Convert model from Jax to TensorFlow
        tf_model = jax2tf.convert(jaxmodel, enable_xla=compiler_cfg.enable_xla_jax_convert)
        tf_func = tf.function(tf_model, autograph=False, jit_compile=True)
        # Get graph definition
        tf_func = tf_func.get_concrete_function(*inputs)
        graph_def = tf_func.graph.as_graph_def(add_shapes=True)

    # Extract flatten inputs
    flattened_inputs, flattened_input_names, _, _ = extract_flatten_inputs(
        framework="jax",
        model=tf_func,
        inputs=inputs,
    )

    m = hashlib.sha256()
    m.update(str(graph_def).encode("utf-8"))
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, m.hexdigest(), framework="jax")
    if cached_graphs is not None:
        return cached_graphs, flattened_inputs

    outputs = [output.name for output in tf_func.outputs]
    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, layout="NCHW", outputs=outputs)
    mod = tvm.transform.Sequential([tvm.relay.transform.Inline()])(mod)

    # Write Graph to the TensorBoard
    # writer = tf.summary.create_file_writer("generated_modules/tensorboard/jax")
    # with writer.as_default():
    #     tf.summary.graph(tf_fun.graph)

    # Construct TVM IR
    mod, param_name_lookup = construct_tvm_ir(
        framework="jax",
        model=jaxmodel,
        tvm_mod=mod,
        params=params,
        compiler_cfg=compiler_cfg,
    )

    # Construct NumPy inputs
    np_inputs = {i: None if x is None else x.numpy() for i, x in zip(flattened_input_names, flattened_inputs)}

    # Compile TVM for Forge
    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        np_inputs,
        framework_outputs,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Extract Graphs (TT, CPU, ...)
    def flatten_params(params, parent_key="", sep="."):
        items = []
        for key, val in params.items():
            new_key = parent_key + sep + key if parent_key else key

            if isinstance(val, MutableMapping):
                items.extend(flatten_params(val, new_key, sep=sep).items())
            else:
                items.append((new_key, val))

        return dict(items)

    # if isinstance(jaxmodel, FlaxPreTrainedModel):
    #     model_params = jaxmodel.params
    # else:
    #     model_params = {}
    #     if hasattr(jaxmodel, 'params'):
    #         model_params = jaxmodel.variables['params']._dict

    model_params = {}
    if hasattr(jaxmodel, "params"):
        model_params = jaxmodel.variables["params"]._dict

    weight_names = list(flatten_params(model_params).keys())
    json_graphs = extract_graphs(
        partitioned_mod, forge_params, flattened_input_names, weight_names, param_name_lookup, graph_hash=m.hexdigest()
    )

    return json_graphs, flattened_inputs


def compile_tf_for_forge(tfmod, *inputs, graph_name, compiler_cfg, verify_cfg=None):
    # Extract framework model outputs
    framework_outputs = extract_framework_model_outputs(
        framework="tensorflow",
        model=tfmod,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
    )

    # Trace module & get graph definition
    @tf.function
    def trace(*inputs):
        kwargs = {}
        import inspect

        arg_names = inspect.getfullargspec(tfmod.call).args
        if "return_dict" in arg_names:
            kwargs["return_dict"] = False

        if "training" in arg_names:
            kwargs["training"] = False
        return tfmod(*inputs, **kwargs)

    full_model = trace.get_concrete_function(*inputs)
    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

    # Extract flatten inputs
    flattened_inputs, flattened_input_names, _, _ = extract_flatten_inputs(
        framework="tensorflow",
        model=frozen_func,
        inputs=inputs,
    )

    m = hashlib.sha256()
    m.update(str(graph_def).encode("utf-8"))
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, m.hexdigest(), framework="tensorflow")
    if cached_graphs is not None:
        return cached_graphs, flattened_inputs

    flattened_outputs = flatten_structured_output([full_model.structured_outputs])
    # Generate TVM module
    outputs = [x.name for x in flattened_outputs]
    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, outputs=outputs)
    mod = tvm.transform.Sequential([tvm.relay.transform.Inline()])(mod)
    record_execution_phase_and_stage(ExecutionStage.TVM_GENERATE_RELAY_IRMODULE)

    # Construct TVM IR
    mod, param_name_lookup = construct_tvm_ir(
        framework="tensorflow",
        model=tfmod,
        tvm_mod=mod,
        params=params,
        compiler_cfg=compiler_cfg,
    )

    # Construct NumPy inputs
    np_inputs = {i: None if x is None else x.numpy() for i, x in zip(flattened_input_names, flattened_inputs)}

    # Compile TVM for Forge
    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        np_inputs,
        framework_outputs,
        input_names=flattened_input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Extract Graphs (TT, CPU, ...)
    weight_names = [weight.name for weight in tfmod.weights]
    json_graphs = extract_graphs(
        partitioned_mod, forge_params, flattened_input_names, weight_names, param_name_lookup, graph_hash=m.hexdigest()
    )

    return json_graphs, flattened_inputs


# TODO: Verify graphdef output vs. TVM output
def compile_tf_graphdef_for_forge(
    graph_def,
    *inputs,
    graph_name,
    compiler_cfg,
):
    output_list_ = compiler_cfg.framework_model_output_names
    # framework_outputs = tfmod(*inputs)
    # if not isinstance(framework_outputs, (list, tuple)):
    #     framework_outputs = [framework_outputs]

    # supported_outputs = (tf.Tensor, torch.Tensor)
    # framework_outputs = [x.numpy() for x in framework_outputs if isinstance(x, supported_outputs)]

    # @tf.function
    # def trace(*inputs):
    #     return tfmod(*inputs, training=False)

    # full_model = trace.get_concrete_function(*inputs)

    # frozen_func = convert_variables_to_constants_v2(full_model)
    # graph_def = frozen_func.graph.as_graph_def()
    input_names = []
    for node in graph_def.node:
        if "input" in node.name and node.op == "Placeholder":
            input_names.append(node.name)

    m = hashlib.sha256()
    m.update(str(graph_def).encode("utf-8"))
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, m.hexdigest(), framework="tf_graphdef")
    if cached_graphs is not None:
        return cached_graphs

    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, layout="NCHW", outputs=output_list_)
    mod = tvm.transform.Sequential([tvm.relay.transform.Inline()])(mod)

    assert (
        compiler_cfg.enable_tvm_constant_prop == True
    ), "Forge Compile only support tf graphdef model with TVM parameter binding."
    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot use constant prop without converting framework params to relay"
        if len(compiler_cfg.tvm_constnat_prop_mask):
            propped_params = {
                k: (v, True)
                for k, v, in params.items()
                if any([mask in k for mask in compiler_cfg.tvm_constnat_prop_mask])
            }
        else:
            propped_params = {k: (v, True) for k, v, in params.items()}
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    target = "llvm"
    mod, params = compile_for_forge(mod, target=target, params=params, graph_name=graph_name)

    # Reconstruct Ops + export forge graph
    partitioned_mod, forge_params = tvm.relay.op.contrib.forge.partition_for_forge(
        mod, graph_name=graph_name, compiler_cfg=compiler_cfg, input_names=input_names
    )

    tvm.relay.build_module.build(partitioned_mod, target=target, params=params)

    json_graphs = extract_graphs(partitioned_mod, forge_params, input_names, [], graph_hash=m.hexdigest())

    return json_graphs


def compile_mxnet_for_forge(module, *inputs, graph_name, compiler_cfg, verify_cfg=None):
    framework_outputs = []
    if verify_cfg is not None and verify_cfg.verify_tvm_compile:
        framework_outputs = module(*inputs)
        if not isinstance(framework_outputs, (list, tuple)):
            framework_outputs = [framework_outputs]

        framework_outputs = [x.asnumpy() for x in framework_outputs if isinstance(x, mx.ndarray.ndarray.NDArray)]

    input_dict = {"input_" + str(i): inp.shape for i, inp in enumerate(inputs)}

    mod_inputs = []
    if isinstance(module, mx.gluon.HybridBlock):
        for name in input_dict:
            mod_inputs.append(mx.sym.Variable(name))
        sym = module(*mod_inputs)
        if isinstance(sym, (list, tuple)):
            sym = mx.sym.Group(sym)
    else:
        sym = module
    graph_string = sym.tojson().encode("utf-8")
    m = hashlib.sha256()
    m.update(graph_string)
    cached_graphs = load_serialized_tvm_graph(compiler_cfg, m.hexdigest(), framework="mxnet")
    if cached_graphs is not None:
        return cached_graphs

    input_name_to_tensor = {name: tensor.asnumpy() for name, tensor in zip(input_dict.keys(), inputs)}
    mod, params = relay.frontend.from_mxnet(module, shape=input_dict)

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot use constant prop without converting framework params to relay"
        propped_params = {k: (v, True) for k, v, in params.items()}
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    _, forge_params = compile_tvm_for_forge(
        mod,
        params,
        input_name_to_tensor,
        framework_outputs,
        input_names=list(input_dict.keys()),
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    dev_json_graph["hash"] = m.hexdigest()
    dev_json_graph["params"] = {}
    for function_name in forge_params.keys():
        dev_json_graph["params"].update(
            {
                name: v.numpy()
                for (k, v), name in zip(
                    forge_params[function_name].items(), dev_json_graph["param_names"][function_name]
                )
            }
        )

    dev_functions = list(dev_json_graph["functions"].keys())
    dev_json_graph["graph"] = dev_json_graph["functions"][dev_functions[0]]

    json_graph = []
    json_graph.append(copy.deepcopy(clean_names(json_graph=dev_json_graph, forge_params=forge_params)))
    return json_graph


def format_tvm_graph_weights(inputs, module, compiler_cfg, framework=None):
    """
    Formats model weights based on specific framework.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations

    Returns
    -------
    OrderedDict, Boolean, Tuple
        Weights, Constant evaluation, Input tensors
    """
    if framework == "pytorch":
        if compiler_cfg.enable_training:
            for param in module.parameters():
                param.requires_grad = True

        torch_weights = module.state_dict()
        named_buffers = dict(module.named_buffers())
        torch_weights.update(named_buffers)
        named_params = dict(module.named_parameters())
        weights = {
            key: (value, named_params[key].requires_grad if key in named_params else False)
            for key, value in torch_weights.items()
        }
    elif framework == "tensorflow":
        weights = {
            weight.name: (
                torch.Tensor(
                    (
                        tf.cast(weight.value(), tf.float32) if weight.value().dtype.is_floating else weight.value()
                    ).numpy()
                ),
                True,
            )
            for weight in module.weights
        }
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = to_pt_tensors(inputs)  # Maybe we can switch all tensors to numpy?
    elif framework == "tf_graphdef":
        weights = {}
    elif framework == "onnx":
        numpy_weights = [onnx.numpy_helper.to_array(weight) for weight in module.graph.initializer]
        names = [weight.name for weight in module.graph.initializer]
        weights = {
            name: (torch.tensor(weight), issubclass(weight.dtype.type, np.floating))
            for name, weight in zip(names, numpy_weights)
        }

        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(onnx.numpy_helper.to_array(x)) for x in inputs if x is not None]
    elif framework == "mxnet":
        weights = {
            name: (
                torch.Tensor(mx_param.data().asnumpy()),
                issubclass(mx_param.data().asnumpy().dtype.type, np.floating),
            )
            for name, mx_param in module.collect_params().items()
        }
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.asnumpy()) for x in inputs if x is not None]
    elif framework == "jax":

        def flatten_params(params, parent_key="", sep="."):
            items = []
            for key, val in params.items():
                new_key = parent_key + sep + key if parent_key else key

                if isinstance(val, MutableMapping):
                    items.extend(flatten_params(val, new_key, sep=sep).items())
                else:
                    items.append((new_key, val))

            return dict(items)

        # if isinstance(module, FlaxPreTrainedModel):
        #     module_params = module.params
        # else:
        #     module_params = {}
        #     if hasattr(module, 'params'):
        #         module_params = module.variables['params']._dict

        module_params = {}
        if hasattr(module, "params"):
            module_params = module.variables["params"]._dict

        module_params = flatten_params(module_params)

        weights = {}
        for key, value in module_params.items():
            torch_tensor = torch.Tensor(np.array(value))
            weights[key] = (torch_tensor, True)

        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.numpy()) for x in inputs if x is not None]

    elif framework == "tflite":
        weights = {}
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.numpy()) for x in inputs if x is not None]
    else:
        raise RuntimeError(f"Unsupported module type {type(module)}")

    return inputs, weights


def get_auto_path(graph_hash, compiler_cfg, is_load):
    """
    Returns auto cache path based on graph hash and tvm submodule git commit

    Parameters
    ----------
    graph_hash: String
        graph hash string

    compiler_cfg: CompilerConfig
        Compiler configurations

    is_load: Boolean
        Are we looking for load path (otherwise store)

    Returns
    -------
    String
        path to store/load graph
    """

    auto_cache = int(os.environ.get("FORGE_ENABLE_TVM_CACHE", 0))
    if bool(auto_cache) and compiler_cfg.tvm_graph_store_path == "" and compiler_cfg.tvm_graph_load_path == "":
        assert (
            auto_cache == -1 or auto_cache == 1
        ), f"FORGE_ENABLE_TVM_CACHE value of {auto_cache} not understood. Set to 1 to enable cache, 0 to disable and -1 to recache module"
        if auto_cache == -1 and is_load:
            auto_path = ""
        else:
            tvm_path = os.path.dirname(tvm.__file__)
            if "site-packages" in tvm_path:
                tvm_info = subprocess.check_output(['pip', 'show', 'tvm']).decode('utf-8')
                for line in tvm_info.split('\n'):
                    if "Version:" in line:
                        tvm_ver = line.split("Version: ")[-1]
                        tvm_hash = tvm_ver.split('+')[-1]
                        break
            else:
                tvm_hash = (
                    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=tvm_path).decode("utf-8").strip()
                )
            auto_path = "generated_modules/tvm_cache/" + tvm_hash + "_" + graph_hash
    else:
        auto_path = compiler_cfg.tvm_graph_load_path if is_load else compiler_cfg.tvm_graph_store_path

    return auto_path


def load_serialized_tvm_graph(compiler_cfg, graph_hash, framework):
    """
    Loads serialized TVM graph representation ported to Forge in form of python dictionary.

    Parameters
    ----------
    compiler_cfg: CompilerConfig
        Compiler configurations

    graph_hash: String
        graph hash string

    Returns
    -------
    Dictionary
        Deserialized TVM graph
    """

    load_path = get_auto_path(graph_hash, compiler_cfg, True)

    if (
        load_path == ""
        or not file_exists(load_path)
        or (compiler_cfg.enable_tvm_constant_prop and framework == "pytorch")
    ):
        return None

    with open(load_path, "r") as file:
        serialized_graph_str = json.load(file)

    json_graphs = []
    for id, json_graph in serialized_graph_str.items():
        serialized_dict = {}
        serialized_dict["graph"] = json.dumps(json_graph["graph"])
        serialized_dict["hash"] = json.dumps(json_graph["hash"])
        serialized_dict["params"] = json_graph["params"]
        serialized_dict["device"] = json_graph["device"]
        if "nid_to_input_idx" in json_graph.keys():
            serialized_dict["nid_to_input_idx"] = {int(k): v for k, v in json_graph["nid_to_input_idx"].items()}
        json_graphs.append(serialized_dict)

    logger.info(f"Successfully loaded serialized TVM graph from {load_path} path")

    return json_graphs


def serialize_and_store_tvm_graph(json_graphs, compiler_cfg, framework):
    """
    Serializes TVM graph representation ported to Forge in form of JSON and stores it
    on the desired destination.

    Parameters
    ----------
    json_graph: Dictionary
        Previously compiled TVM graph pored to Forge representation

    compiler_cfg: CompilerConfig
        Compiler configurations

    Returns
    -------
    """

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    graph_hash = json_graphs[0]["hash"]
    store_path = get_auto_path(graph_hash, compiler_cfg, False)
    if (
        store_path == ""
        or (compiler_cfg.enable_tvm_constant_prop and framework == "pytorch")
        or not len(dev_json_graph["graph"])
    ):
        return

    serilized_dict = {}

    for id, json_graph in enumerate(json_graphs):
        graph_dict = json.loads(json_graph["graph"])
        params_dict = json_graph["params"]
        serilized_dict[str(id)] = {}
        serilized_dict[str(id)]["graph"] = graph_dict
        serilized_dict[str(id)]["params"] = params_dict
        serilized_dict[str(id)]["device"] = json_graph["device"]
        serilized_dict[str(id)]["hash"] = json_graph["hash"]
        if "nid_to_input_idx" in json_graph.keys():
            serilized_dict[str(id)]["nid_to_input_idx"] = json_graph["nid_to_input_idx"]

    serilized_str = json.dumps(serilized_dict, cls=NumpyArrayEncoder, indent=2)

    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as file:
        file.write(serilized_str)

    logger.info(f"Successfully stored serilized TVM graph to {store_path} path")
