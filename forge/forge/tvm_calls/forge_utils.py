# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict
from collections.abc import MutableMapping
import inspect
import os

import paddle
import torch
import flax
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from transformers import FlaxPreTrainedModel
from transformers.utils.generic import ModelOutput as HFModelOutput
from transformers.modeling_outputs import ModelOutput

import tvm
from tvm.relay import ExprVisitor
from forge.config import CompilerConfig
from forge.tvm_utils import flatten_inputs, flatten_structured_output
from forge.tensor import to_pt_tensors, to_pd_tensors
from forge.tvm_calls.relay.op.forge import extract_function_callnodes, trace_to_origin
from typing import Optional


def extract_framework_model_outputs(
    framework: str,
    model,
    inputs,
    verify_tvm_compile: bool = False,
    input_dict={},
    onnx_path: Optional[str] = None,
):
    framework_outputs = []

    if verify_tvm_compile:
        return framework_outputs

    if framework == "pytorch" or framework == "paddle":
        assert model.training == False

        framework_outputs = model(*inputs)
        if isinstance(framework_outputs, ModelOutput):
            framework_outputs = framework_outputs.to_tuple()

        if not isinstance(framework_outputs, (list, tuple)):
            if isinstance(framework_outputs, (torch.Tensor, paddle.Tensor)):
                framework_outputs = [framework_outputs]
            elif isinstance(framework_outputs, OrderedDict):
                framework_outputs = tuple(framework_outputs.values())
            else:
                assert False, "Don't know what to do with this"
        elif any([isinstance(x, (tuple, list)) for x in framework_outputs]):

            def flatten_outputs(outputs):
                new_outputs = []
                if isinstance(outputs, (tuple, list)):
                    for output in outputs:
                        new_outputs.extend(flatten_outputs(output))
                else:
                    new_outputs.append(outputs)
                return new_outputs

            framework_outputs = flatten_outputs(framework_outputs)

        framework_outputs = [x.detach().numpy() for x in framework_outputs]

    elif framework == "tensorflow":
        kwargs = {}
        import inspect

        arg_names = inspect.getfullargspec(model.call).args
        if "return_dict" in arg_names:
            kwargs["return_dict"] = False

        if "training" in arg_names:
            kwargs["training"] = False

        framework_outputs = model(*inputs, **kwargs)

        # TODO ref sha: 1fe78625c809e6ca887a8da5fdde44836830f990
        # Figure out how to sort dictionary outputs:
        #
        # if isinstance(framework_outputs, dict):
        #     framework_outputs = list(framework_outputs.values())
        if isinstance(framework_outputs, dict):
            framework_outputs = list(framework_outputs.values())
        if not isinstance(framework_outputs, (list, tuple)):
            framework_outputs = [framework_outputs]

        framework_outputs = flatten_structured_output(framework_outputs)
        supported_outputs = (tf.Tensor, torch.Tensor)
        framework_outputs = [x.numpy() for x in framework_outputs if isinstance(x, supported_outputs)]

    elif framework == "jax":
        import jax.numpy as jnp

        args = [
            jnp.asarray(
                x.numpy(),
            )
            for x in inputs
        ]
        framework_outputs = model(*args)
        if isinstance(framework_outputs, HFModelOutput):
            framework_outputs = list(framework_outputs.values())

        if not isinstance(framework_outputs, (list, tuple)):
            framework_outputs = [framework_outputs]

        outputs = to_pt_tensors(framework_outputs)
        outputs = flatten_structured_output([outputs])
        framework_outputs = [x.detach().numpy() for x in outputs]
    elif framework == "onnx":
        output_names = []
        for out in model.graph.output:
            output_names.append(out.name)

        so = ort.SessionOptions()
        so.inter_op_num_threads = 2
        so.intra_op_num_threads = 2

        if onnx_path is not None:
            onnx_model = onnx_path
        else:
            onnx_model = model.SerializeToString()
        ort_sess = ort.InferenceSession(onnx_model, sess_options=so)
        framework_outputs = ort_sess.run(output_names, input_dict)

    elif framework == "tflite":
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.allocate_tensors()
        model.set_tensor(input_details[0]["index"], *inputs)
        model.invoke()
        framework_outputs = model.get_tensor(output_details[0]["index"])

    else:
        raise RuntimeError("Unsupported framework type: {}".format(framework))

    return framework_outputs


def extract_flatten_inputs(framework: str, model, inputs, input_names=[]):
    if framework == "pytorch":
        if len(input_names) == 0:
            input_names = [i.debugName().replace(".", "_") for i in list(model.graph.inputs())[1:]]
        else:
            graph_input_names = [i.debugName() for i in list(model.graph.inputs())[1:]]
            assert len(graph_input_names) == len(input_names), "Number of input names must match number of inputs."
            for graph_name, input_name in zip(graph_input_names, input_names):
                assert input_name in graph_name, "Input names must match graph input names."

        def get_input_structure(inputs, input_names):
            input_structure = []
            if isinstance(inputs, (list, tuple)):
                for i in range(len(inputs)):
                    input = inputs[i]
                    if isinstance(input, (list, tuple)):
                        sub_names = [input_names[i] + "_" + str(ii) for ii in range(len(input))]
                        structure = (input_names[i], get_input_structure(input, sub_names))
                    elif isinstance(input, dict):
                        structure = (input_names[i], {k: v.shape for k, v in input.items()})
                    else:
                        structure = (input_names[i], (tuple(input.shape), str(input.dtype).replace("torch.", "")))
                    input_structure.append(tuple(structure))
            else:
                input_structure = OrderedDict()
                for k, v in inputs.items():
                    input_structure[k] = (tuple(v.shape), str(input.dtype).replace("torch.", ""))
            return input_structure

        input_structure = get_input_structure(inputs, input_names)

        flattened_inputs, flattened_input_names, flattened_name_map = flatten_inputs(inputs, input_names)

    elif framework == "paddle":
        paddle_inputs = to_pd_tensors(inputs)
        input_structure = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in paddle_inputs]

        if hasattr(model, "_input_args_names"):
            flattened_input_names = model._input_args_names
        else:
            # TODO: Find a better way to get input names for paddle models
            # When the forward function has optional parameters, we assume they are provided in order in the input,
            # so we use the first len(inputs) elements of the forward function signature as the input names
            flattened_input_names = list(inspect.signature(model.forward).parameters.keys())[: len(inputs)]

        flattened_inputs, _, flattened_name_map = flatten_inputs(inputs, flattened_input_names)

    elif framework == "tensorflow":
        # The tensorflow trace automatically flattens inputs
        flattened_inputs, _, _ = flatten_inputs(inputs)
        flattened_input_names = [tensor.name.split(":")[0] for tensor in model.inputs]
        flattened_name_map = None
        input_structure = None

    elif framework == "jax":
        # The tensorflow trace automatically flattens inputs
        flattened_inputs, _, _ = flatten_inputs(inputs)
        flattened_input_names = [tensor.name.split(":")[0] for tensor in model.inputs]

        flattened_name_map = None
        input_structure = None

    else:
        raise RuntimeError("Unsupported framework type: {}".format(framework))

    return flattened_inputs, flattened_input_names, flattened_name_map, input_structure


def construct_tvm_ir(framework: str, model, tvm_mod, params, compiler_cfg: CompilerConfig):
    if framework == "pytorch" or framework == "paddle":
        param_name_lookup = {}

        if not compiler_cfg.enable_tvm_constant_prop:
            tvm_mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], {}))
        else:
            if len(compiler_cfg.tvm_constnat_prop_mask):
                propped_params = {
                    k: (v, True)
                    for k, v, in params.items()
                    if any([mask in k for mask in compiler_cfg.tvm_constnat_prop_mask])
                }
            else:
                propped_params = {k: (v, True) for k, v, in params.items()}
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], propped_params)
            )

    elif framework == "tensorflow":
        # TODO: Destupidify this! (Maybe we can sort by a substring of the weight names to make this more efficient)
        found_weights = []
        param_name_lookup = {}
        non_weight_params = {}  # Some parameters (like causal mask) are not weights

        for (bad_name, value) in params.items():
            weight_found = False
            for tf_weight in model.weights:
                if np.array_equal(tf_weight.value().numpy(), value.numpy()) and tf_weight.name not in found_weights:
                    param_name_lookup[bad_name] = tf_weight.name
                    weight_found = True
                    found_weights.append(tf_weight.name)
                    break
            if not weight_found:
                param_name_lookup[bad_name] = bad_name
                non_weight_params[bad_name] = (value, False)

        if not compiler_cfg.enable_tvm_constant_prop:
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], non_weight_params)
            )
        else:
            if len(compiler_cfg.tvm_constnat_prop_mask):
                propped_params = {
                    k: (v, True)
                    for k, v, in params.items()
                    if any([mask in param_name_lookup[k] for mask in compiler_cfg.tvm_constnat_prop_mask])
                }
                propped_params.update(non_weight_params)
            else:
                propped_params = {k: (v, True) for k, v in params.items()}
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], propped_params)
            )

    elif framework == "jax":

        def flatten(d, parent_key="", sep="."):
            """
            Flatten a nested dictionary structure into a single level dictionary
            with concatenated keys:
            Example: {"a": {"b": 1, "c": 2}} -> {"a.b": 1, "a.c": 2}
            """
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        found_weights = set()
        param_name_lookup = {}
        non_weight_params = {}
        model_params = {}

        # Extract model parameters based on model type.
        # Handle different JAX model types:
        # - FlaxPreTrainedModel: model.params
        # - flax.linen.Module: model.variables["params"]
        if hasattr(model, "params"):
            model_params = model.params
        elif hasattr(model, "variables") and "params" in model.variables:
            model_params = model.variables["params"]

        # Flatten the model parameters into a single level dictionary.
        # And convert model parameters to a set of numpy arrays.
        model_params = flatten(model_params)
        model_params_set = {name: np.asarray(jax_value) for name, jax_value in model_params.items()}

        # Map the model parameters to the corresponding parameters in the params dictionary.
        for bad_name, value in params.items():
            value_np = value.numpy()
            # Find matching parameter by comparing array values.
            # If match found, creating a mapping between the bad_name and the matched_name.
            # If no match found, this parameter is not a weight.
            matched_name = next(
                (
                    name
                    for name, jax_value in model_params_set.items()
                    if name not in found_weights and np.array_equal(jax_value, value_np)
                ),
                None,
            )
            if matched_name:
                param_name_lookup[bad_name] = matched_name
                found_weights.add(matched_name)
            else:
                param_name_lookup[bad_name] = bad_name
                non_weight_params[bad_name] = value

        if not compiler_cfg.enable_tvm_constant_prop:
            # If constant propagation is disabled, bind the non-weight parameters to the TVM module.
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], non_weight_params)
            )
        else:
            # If constant propagation is enabled, bind the parameters based on mask settings.
            if len(compiler_cfg.tvm_constnat_prop_mask):
                propped_params = {
                    k: v
                    for k, v, in params.items()
                    if any([mask in param_name_lookup[k] for mask in compiler_cfg.tvm_constnat_prop_mask])
                }
                propped_params.update(non_weight_params)
            else:
                propped_params = params
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], propped_params)
            )
    else:
        raise RuntimeError(f"Unsupported framework type: {framework}")

    return tvm_mod, param_name_lookup


def has_op(module, opname, attrs={}):
    class Visitor(ExprVisitor):
        def __init__(self):
            super().__init__()
            self.has_op = False

        def visit_call(self, call):
            if call.op.name == opname:
                self.has_op = True
                for key in attrs.keys():
                    self.has_op &= key in call.attrs.keys() and call.attrs[key] == attrs[key]
                if self.has_op:
                    return
            super().visit_call(call)

    visitor = Visitor()
    visitor.visit(module)
    return visitor.has_op


def paddle_trace(paddlemod, input_spec=None, inputs=None):
    """
    Converts paddle.nn.Layer to paddle.nn.TranslatedLayer needed for TVM compilation.
    """
    assert input_spec is not None or inputs is not None, "Either input_spec or inputs must be provided."

    if input_spec is None and inputs is not None:
        # Convert inputs to paddle static input spec
        input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]

    traced_model = paddle.jit.to_static(paddlemod, input_spec=input_spec, full_graph=True)

    # Model must be saved and loaded in order to have TranslatedLayer type which is needed for the next step
    model_save_path = "traced_model_tmp"
    paddle.jit.save(traced_model, model_save_path)
    loaded_model = paddle.jit.load(model_save_path)

    # Parameter names are not preserved in the loaded model, so we need to map them back to the original model
    original_param_names = sorted([param.name for param in paddlemod.parameters()])
    new_param_names = sorted([param.name for param in loaded_model.parameters()])
    param_name_lookup = {new: old for new, old in zip(new_param_names, original_param_names)}

    # Clean up
    for ext in [".pdiparams", ".pdiparams.info", ".pdmodel"]:
        file = f"{model_save_path}{ext}"
        if os.path.exists(file):
            os.remove(file)

    return loaded_model, param_name_lookup
