# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from tvm.relay.dataflow_pattern import *
import tvm
from tvm import relay
from tvm.contrib import graph_executor

from loguru import logger


def get_relay_output(mod, params, inputs, target):
    # Build and Run Relay modules with inputs as (key : tensor) pair
    # Then, inputs dont need to be in the same order as 'mod' defines.
    ret_type = mod["main"].checked_type.ret_type
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build_module.build(mod, target=target, params=params)
    m = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    m.run(**inputs)

    def _unflatten(flat_iter, cur_type):
        import tvm.relay.ty as _ty

        if isinstance(cur_type, _ty.TensorType):
            return next(flat_iter)
        if isinstance(cur_type, _ty.TupleType):
            fields = []
            for field_type in cur_type.fields:
                field = _unflatten(flat_iter, field_type)
                fields.append(field)
            return fields
        raise ValueError("Return type", ret_type, "contains unsupported type", cur_type)

    flattened = []
    import tvm.runtime.ndarray as _nd

    for i in range(m.get_num_outputs()):
        flattened.append(m.get_output(i).copyto(_nd.cpu(0)))
    relay_outputs = _unflatten(iter(flattened), ret_type)

    if not isinstance(relay_outputs, (list, tuple)):
        relay_outputs = [relay_outputs]
    relay_outputs = [x.numpy() for x in flattened]
    return relay_outputs


def verify_outputs(framework_outputs, relay_outputs, compile_location, rtol=1e-02, atol=1e-04, pcc=None):
    allowed_to_fail = False
    if len(framework_outputs) != len(relay_outputs):
        logger.error(
            f"Different number of outputs. Framework: {len(framework_outputs)}, TVM: {len(relay_outputs)} after {compile_location}"
        )

    for i, (fr_out, tvm_out) in enumerate(zip(framework_outputs, relay_outputs)):
        if fr_out.shape != tvm_out.shape:
            logger.error(
                f"Different shapes for outputs. Framework: {fr_out.shape}, TVM: {tvm_out.shape} after {compile_location}"
            )

        if pcc is None:
            ok = np.allclose(fr_out, tvm_out, rtol=rtol, atol=atol, equal_nan=True)
        else:
            pcc_value = np.min(
                np.ma.corrcoef(np.ma.masked_invalid(fr_out.flatten()), np.ma.masked_invalid(tvm_out.flatten()))
            )
            if isinstance(pcc_value, np.ma.core.MaskedConstant):
                pcc_value = 1.0
            ok = pcc_value >= pcc

        if not ok:
            logger.error(f"Tensor mismatch on output {i} between framework and TVM after {compile_location}.")
            logger.trace(f"Framework: (shape = {fr_out.shape}")
            logger.trace(fr_out)
            logger.trace(f"TVM: (shape = {tvm_out.shape}")
            logger.trace(tvm_out)
            logger.info(
                "Max ATOL Delta: "
                + "{:.3e}".format(np.max(np.abs((fr_out - tvm_out))).item())
                + ", atol="
                + "{}".format(atol)
            )
            logger.info(
                "Max RTOL Delta: "
                + "{:.3e}".format(np.max(np.abs((fr_out - tvm_out)) / tvm_out).item())
                + ", rtol="
                + "{}".format(rtol)
            )
            if pcc is not None:
                logger.info(f"PCC got={pcc_value}, required={pcc}")
            if not allowed_to_fail:
                raise RuntimeError

    logger.info(f"Verified TVM Relay outputs against framework outputs after {compile_location}")


def verify_tvm_compile(mod, params, inputs, target, framework_outputs, compile_location, verify_cfg=None):
    relay_outputs = get_relay_output(mod, params, inputs, target)

    # Verify compile passes (original relay passes + forge passes)
    if verify_cfg:
        verify_outputs(
            framework_outputs,
            relay_outputs,
            compile_location,
            rtol=verify_cfg.rtol,
            atol=verify_cfg.atol,
            pcc=verify_cfg.pcc,
        )
    else:
        verify_outputs(framework_outputs, relay_outputs, compile_location)


def is_unsqueeze(call):
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    remove_one_from_output_shape = [x for x in output_shape if x != 1]

    joint_size = min(len(input_shape), len(remove_one_from_output_shape))

    superfluous_reshape = all(
        [input_shape[i] == remove_one_from_output_shape[i] for i in range(-1, -1 * joint_size - 1, -1)]
    )

    if superfluous_reshape and len(input_shape) < len(output_shape):
        return True

    return False


def is_squeeze(call):
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    joint_size = min(len(input_shape), len(output_shape))

    superfluous_reshape = all([input_shape[i] == output_shape[i] for i in range(-1, -1 * joint_size - 1, -1)])

    if superfluous_reshape and len(input_shape) > len(output_shape):
        return True

    return False


def is_superfluous_reshape(call):
    assert call.op.name == "reshape"
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    joint_size = min(len(input_shape), len(output_shape))

    superfluous_reshape = all([input_shape[i] == output_shape[i] for i in range(-1, -1 * joint_size - 1, -1)])

    if len(input_shape) > len(output_shape):
        extra_dims = len(input_shape) - len(output_shape)
        if all([extra_dim == 1 for extra_dim in input_shape[:extra_dims]]):
            superfluous_reshape = superfluous_reshape and True
    elif len(output_shape) > len(input_shape):
        extra_dims = len(output_shape) - len(input_shape)
        if all([extra_dim == 1 for extra_dim in output_shape[:extra_dims]]):
            superfluous_reshape = superfluous_reshape and True

    return superfluous_reshape


def match_einsum_pattern(pattern, query):
    query = query.replace(" ", "")
    pattern = pattern.replace(" ", "")
    if len(query) != len(pattern):
        return False

    query_dict = {}
    for char in query:
        query_dict[char] = []

    for i in range(len(query)):
        char = query[i]
        query_dict[char].append(i)

    pattern_dict = {}
    for char in pattern:
        pattern_dict[char] = []

    for i in range(len(pattern)):
        char = pattern[i]
        pattern_dict[char].append(i)

    return sorted(list(query_dict.values())) == sorted(list(pattern_dict.values()))
