# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import re
import json
from collections import OrderedDict
from typing import Dict, List
from enum import Enum

from loguru import logger

import torch
import numpy as np
import pytest

# import forge._C.pattern_matcher as pypattern_matcher
from forge.module import OnnxModule, ForgeModule, TFLiteModule
from forge.config import _get_global_compiler_config
from forge.verify.config import _get_global_verify_config
import forge
from forge.tensor import to_pt_tensors
from forge.tvm_utils import flatten_inputs

import os
import sys
import importlib

from forge.python_codegen import PyTorchWriter, ForgeWriter, PythonWriter, pytorch_df_str_from_str
from forge.utils import create_excel_file


def populate_torch_all_to_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_broadcast_to_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "shape": {
            "val": [-1] + list(curr_node["attrs"]["shape"][0][0])[1:],
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_index_select_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    strides = [int(strides) for strides in curr_node["attrs"]["strides"][0]]
    begin = [int(begin) for begin in curr_node["attrs"]["begin"][0]]
    end = [int(e) for e in curr_node["attrs"]["end"][0]]

    assert len(strides) == 1 and len(begin) == 1 and len(end) == 1, "Strided slice should be on 1 axis"
    assert int(curr_node["attrs"]["num_inputs"]) == 1
    assert len(list(curr_node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axes"][0][0]),
            "inp_pos": 1,
        },
        "index": {
            "val": f"torch.arange({begin[0]}, {end[0]}, {strides[0]})",
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_adv_index_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    curr_node_shape = curr_node["attrs"]["shape"][0][0]
    assert (len(curr_node_shape) == 1) or (
        curr_node_shape[0] == 1 and len(curr_node_shape)
    ), "Only 1D indexing is supported"

    node_to_index = graph["nodes"][int(curr_node["inputs"][0][0])]
    ndim_node_to_index = len(node_to_index["attrs"]["shape"][0][0])

    args["attr"] = {
        "dim": {
            "val": -ndim_node_to_index,
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_reshape_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    output_shape = list(curr_node["forge_shape"])
    output_shape[0] = -1

    args["attr"] = {
        "shape": {
            "val": output_shape,
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_cumsum_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_log_softmax_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_softmax_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_squeeze_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    assert len(curr_node["attrs"]["axis"][0][0]) == 1, "Currently support only single axis for squeeze"

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_scatter_add_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    assert curr_node["attrs"]["reduction"][0][0] == "add", "TODO Add other scatter elements reductions"
    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_scatter_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_cast_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    dtype = curr_node["attrs"]["dtype"][0][0]
    if dtype == "uint1":
        dtype = "bool"

    args["attr"] = {
        "dtype": {
            "val": f"torch.{dtype}",
            "inp_pos": 1,
        }
    }
    args["inplace"] = True

    return args


def populate_torch_transpose_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    # Handle TVM axes stile for PyTorch case. More precisely,
    # TVM defines new operand positions as axes, while PyTorch
    # requires 2 indexes which defines which dimensions should
    # be transposed.
    #
    # E.g.
    # [0, 2, 1, 3] => [1, 2]
    changed_axes = []
    axes = [int(axis) for axis in curr_node["attrs"]["axes"][0]]
    for i, axis in enumerate(axes):
        if i != axis:
            changed_axes.append(str(i))
    curr_node["attrs"]["axes"][0] = changed_axes

    assert len(curr_node["attrs"]["axes"][0]) == 2, "Pytorch Transpose only supports 2 dimensions"

    args["attr"] = {
        "dim0": {
            "val": int(curr_node["attrs"]["axes"][0][0]),
            "inp_pos": 1,
        },
        "dim1": {
            "val": int(curr_node["attrs"]["axes"][0][1]),
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_sum_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    dim = int(curr_node["attrs"]["axis"][0][0])
    keepdims = bool(curr_node["attrs"]["keepdims"][0][0])
    args["attr"] = {"dim": {"val": dim, "inp_pos": 1}, "keepdims": {"val": keepdims, "inp_pos": 2}}

    return args


def populate_torch_argmax_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        },
        "keepdims": {
            "val": bool(curr_node["attrs"]["keepdims"][0][0]),
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_argwhere_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    # args['attr'] = {
    #     "as_tuple": {
    #         "val": True,
    #         "inp_pos": 2,
    #         "as_named": True,
    #     },
    # }

    return args


def populate_torch_reduce_max_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        },
        "keepdims": {
            "val": bool(curr_node["attrs"]["keepdims"][0][0]),
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_concat_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    # Determine dim
    dim = None
    act_shape = graph["nodes"][curr_node["inputs"][0][0]]["forge_shape"]
    curr_node_shape = curr_node["forge_shape"]
    for dim, (i, o) in enumerate(zip(act_shape, curr_node_shape)):
        if i != o:
            dim = int(dim - len(act_shape))
            break
    assert dim, "Could not find concatenate axis"

    args["attr"] = {
        "dim": {
            "val": dim,
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_stack_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        },
    }

    return args


def populate_torch_tile_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)
    args["attr"] = {
        "reps": {
            "val": [int(reps) for reps in curr_node["attrs"]["reps"][0]],
            "inp_pos": 1,
        },
    }

    return args


def populate_torch_layernorm_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    epsilon = float(curr_node["attrs"]["epsilon"][0][0])
    epsilon = round(epsilon, 10)
    input_shape = curr_node["attrs"]["shape"][0][0]
    dim = int(curr_node["attrs"]["axis"][0][0])
    normalized_shape = input_shape[dim]
    args["attr"] = {
        "normalized_shape": {
            "val": (normalized_shape,),
            "inp_pos": 1,
        },
        "epsilon": {
            "val": epsilon,
            "inp_pos": 4,
        },
    }

    return args


def populate_torch_dropout_args(graph, nid, training):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "training": {
            "val": training,
            "inp_pos": 2,
        },
    }
    return args


def _populate_torch_init_args(graph, nid):
    curr_node = graph["nodes"][nid]

    args = {"inp": [], "attr": {}, "inplace": False}
    for i in range(0, int(curr_node["attrs"]["num_inputs"])):
        args["inp"].append(graph["nodes"][curr_node["inputs"][i][0]]["forge_name"])

    return curr_node, args


tvm_to_pytorch_op_map = {
    "abs": "abs",
    "add": "add",
    "all": "all",
    "argmax": "argmax",
    "argwhere": "argwhere",
    "broadcast_to_like": "broadcast_to_like",
    "broadcast_to": "broadcast_to",
    "cast": "cast",
    "concatenate": "concatenate",
    "const": "const",
    "cumsum": "cumsum",
    "embedding": "embedding",
    "equal": "equal",
    "exp": "exp",
    "floor": "floor",
    "greater_equal": "greater_equal",
    "identity": "identity",
    "less_equal": "less_equal",
    "less": "less",
    "logical_and": "logical_and",
    "logical_not": "logical_not",
    "max": "reduce_max",
    "maximum": "maximum",
    "minimum": "minimum",
    "multiply": "multiply",
    "nn.batch_matmul": "matmul",
    "nn.dense": "linear",
    "nn.log_softmax": "log_softmax",
    "nn.matmul": "matmul",
    "nn.softmax": "softmax",
    "not_equal": "not_equal",
    "power": "power",
    "forge_cpudevice.adv_index": "adv_index",
    "forge_cpudevice.concatenate": "concatenate",
    "reciprocal": "reciprocal",
    "reshape": "reshape",
    "scatter_elements": "scatter_add",
    "scatter": "scatter",
    "sigmoid": "sigmoid",
    "sign": "sign",
    "sqrt": "sqrt",
    "squeeze": "squeeze",
    "stack": "stack",
    "strided_slice": "index_select",
    "subtract": "subtract",
    "sum": "sum",
    "take": "embedding",
    "tile": "tile",
    "transpose": "transpose",
    # "take"                        : "take",
    "where": "where",
    "layernorm": "layernorm",
    "forge_cpudevice.dropout": "dropout",
}

pytorch_op_to_function_name = {
    "abs": "torch.abs",
    "add": "torch.add",
    "adv_index": "torch.index_select",
    "all": "torch.all",
    "argmax": "torch.argmax",
    "argwhere": "torch.nonzero",
    "broadcast_to_like": "torch.broadcast_to_like",
    "broadcast_to": "torch.broadcast_to",
    "cast": "to",
    "concatenate": "torch.cat",
    "const": "torch.Tensor",
    "cumsum": "torch.cumsum",
    "embedding": "torch.embedding",
    "equal": "torch.eq",
    "exp": "torch.exp",
    "floor": "torch.floor",
    "greater_equal": "torch.ge",
    "identity": "torch.nn.Identity()",
    "index_select": "torch.index_select",
    "less_equal": "torch.le",
    "less": "torch.less",
    "linear": "torch.nn.functional.linear",
    "log_softmax": "torch.nn.functional.log_softmax",
    "logical_and": "torch.logical_and",
    "logical_not": "torch.logical_not",
    "matmul": "torch.matmul",
    "maximum": "torch.maximum",
    "minimum": "torch.minimum",
    "multiply": "torch.multiply",
    "not_equal": "torch.not_equal",
    "power": "torch.pow",
    "reciprocal": "torch.reciprocal",
    "reduce_max": "torch.max",
    "repeat": "torch.Tensor.repeat",
    "reshape": "torch.reshape",
    "scatter_add": "torch.scatter_add",
    "scatter": "torch.scatter",
    "sigmoid": "torch.sigmoid",
    "sign": "torch.sign",
    "softmax": "torch.nn.functional.softmax",
    "sqrt": "torch.sqrt",
    "squeeze": "torch.squeeze",
    "stack": "torch.stack",
    "subtract": "torch.subtract",
    "sum": "torch.sum",
    "tile": "torch.Tensor.repeat",
    "transpose": "torch.transpose",
    # "take"                          : "torch.take",
    "where": "torch.where",
    "layernorm": "torch.nn.functional.layer_norm",
    "dropout": "torch.nn.functional.dropout",
}

pytorch_ops_needing_arguments = {
    "adv_index": populate_torch_adv_index_args,
    "all": populate_torch_all_to_args,
    "argmax": populate_torch_argmax_args,
    "argwhere": populate_torch_argwhere_args,
    "broadcast_to": populate_torch_broadcast_to_args,
    "cast": populate_torch_cast_args,
    "concatenate": populate_torch_concat_args,
    "cumsum": populate_torch_cumsum_args,
    "index_select": populate_torch_index_select_args,
    "log_softmax": populate_torch_log_softmax_args,
    "reduce_max": populate_torch_reduce_max_args,
    "reshape": populate_torch_reshape_args,
    "scatter_add": populate_torch_scatter_add_args,
    "scatter": populate_torch_scatter_args,
    "softmax": populate_torch_softmax_args,
    "squeeze": populate_torch_squeeze_args,
    "stack": populate_torch_stack_args,
    "sum": populate_torch_sum_args,
    "tile": populate_torch_tile_args,
    "transpose": populate_torch_transpose_args,
    # "power"                         : populate_torch_power_args,
    "layernorm": populate_torch_layernorm_args,
    "dropout": populate_torch_dropout_args,
}


def populate_binary_stack_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    input_shape = graph["nodes"][node["inputs"][0][0]]["forge_shape"]
    node_shape = node["forge_shape"]

    for dim, (i, o) in enumerate(zip(input_shape, node_shape)):
        if i != o:
            args = [
                ("dim", f"{dim - len(input_shape)}"),
            ]
            return args

    return args


def populate_conv2d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    reordered_padding = [
        padding[1],
        padding[3],
        padding[0],
        padding[2],
    ]
    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    groups = int(node["attrs"]["groups"][0][0])
    args.append(
        (
            "groups",
            f"{groups}",
        )
    )

    channel_last = int(node["attrs"]["data_layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_conv3d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [depth_first, top, left, depth_last, bottom, right]
    # Convert to [left right top bottom depth_first depth_last]
    reordered_padding = [padding[2], padding[5], padding[1], padding[4], padding[0], padding[3]]
    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    groups = int(node["attrs"]["groups"][0][0])
    args.append(
        (
            "groups",
            f"{groups}",
        )
    )

    channel_last = int(node["attrs"]["data_layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_cumsum_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    axis = node["attrs"]["axis"][0][0]
    args.append(
        (
            "axis",
            f"{axis}",
        )
    )

    exclusive = node["attrs"]["exclusive"][0][0]
    args.append(
        (
            "exclusive",
            f"{exclusive}",
        )
    )

    return args


def populate_conv2d_transpose_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    args.append(
        (
            "padding",
            f"{padding[0]}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    assert all([x == 1 for x in dilation]), "Only supports dilation of 1"
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    in_channel = None
    for input_ in node["inputs"]:
        input_nid = input_[0]
        input_node = graph["nodes"][input_nid]
        if input_node["op"] == "parameter" and input_node["name"].endswith("weight"):
            in_channel = input_node["attrs"]["shape"][0][0][0]
            break

    groups = int(node["attrs"]["groups"][0][0])
    assert groups == 1 or (in_channel is not None and groups == in_channel), "Only supports group of 1 or in_channel"
    args.append(
        (
            "groups",
            f"{groups}",
        )
    )

    kernel_size = [int(kernel) for kernel in node["attrs"]["kernel_size"][0]]
    channel_last = int(node["attrs"]["data_layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_argmax_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    dim = int(node["attrs"]["axis"][0][0])
    if dim >= 0:
        dim -= len(list(graph["nodes"][nid]["forge_shape"]))

    args = [
        ("dim", f"{dim}"),
    ]
    return args


def populate_avgpool3d_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    args.append(
        (
            "kernel_size",
            f"{kernel_size}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [depth_first, top, left, depth_last, bottom, right]
    # Convert to [left right top bottom depth_first depth_last]
    reordered_padding = [padding[2], padding[5], padding[1], padding[4], padding[0], padding[3]]

    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])  # 1 for True
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    count_include_pad = int(node["attrs"]["count_include_pad"][0][0])
    count_include_pad = "True" if count_include_pad == 1 else "False"
    args.append(("count_include_pad", count_include_pad))

    channel_last = int(node["attrs"]["layout"][0][0] == "NDHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_avgpool2d_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    args.append(
        (
            "kernel_size",
            f"{kernel_size}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    reordered_padding = [
        padding[1],
        padding[3],
        padding[0],
        padding[2],
    ]
    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    # dilation = [int(dilate) for dilate in node["attrs"]["dilation"][0]]
    # attrs.append(dilation[0])

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])  # 1 for True
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    count_include_pad = int(node["attrs"]["count_include_pad"][0][0])
    count_include_pad = "True" if count_include_pad == 1 else "False"
    args.append(("count_include_pad", count_include_pad))

    channel_last = int(node["attrs"]["layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_avgpool1d_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    args.append(
        (
            "kernel_size",
            f"{kernel_size}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    args.append(
        (
            "padding",
            f"{padding}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])  # 1 for True
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    count_include_pad = int(node["attrs"]["count_include_pad"][0][0])
    count_include_pad = "True" if count_include_pad == 1 else "False"
    args.append(("count_include_pad", count_include_pad))
    return args


def populate_maxpool1d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    assert len(kernel_size) == 1
    args.append(
        (
            "kernel_size",
            f"{kernel_size[0]}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert len(strides) == 1
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    assert all([p == padding[0] for p in padding])
    args.append(
        (
            "padding",
            f"{padding[0]}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert len(dilation) == 1
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    return args


def populate_maxpool2d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    assert all([dim == kernel_size[0] for dim in kernel_size])
    args.append(
        (
            "kernel_size",
            f"{kernel_size[0]}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    reordered_padding = [
        padding[1],
        padding[3],
        padding[0],
        padding[2],
    ]
    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    args.append(("max_pool_add_sub_surround", f"{compiler_cfg.max_pool_add_sub_surround}"))
    args.append(("max_pool_add_sub_surround_value", f"{compiler_cfg.max_pool_add_sub_surround_value}"))

    channel_last = int(node["attrs"]["layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_maxpool3d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    assert all([dim == kernel_size[0] for dim in kernel_size])
    args.append(
        (
            "kernel_size",
            f"{kernel_size[0]}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [depth_first, top, left, depth_last, bottom, right]
    # Convert to [left right top bottom depth_first depth_last]
    reordered_padding = [padding[2], padding[5], padding[1], padding[4], padding[0], padding[3]]
    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    args.append(("max_pool_add_sub_surround", f"{compiler_cfg.max_pool_add_sub_surround}"))
    args.append(("max_pool_add_sub_surround_value", f"{compiler_cfg.max_pool_add_sub_surround_value}"))

    channel_last = int(node["attrs"]["layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_vstack_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    output_shape = node["attrs"]["shape"][0][0]

    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    slice_size = input_shape[-3] // output_shape[-3]
    args = [("slices", f"{slice_size}")]
    return args


def populate_unsqueeze_args(graph, nid, compiler_cfg):
    dim = graph["nodes"][nid]["attrs"]["axis"][0][0]
    args = [("dim", f"{dim}")]
    return args


def populate_squeeze_args(graph, nid, compiler_cfg):
    dim = graph["nodes"][nid]["attrs"]["axis"][0][0]
    args = [("dim", f"{dim}")]
    return args


def populate_vslice_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    output_shape = node["attrs"]["shape"][0][0]

    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    slice_size = output_shape[-3] // input_shape[-3]
    args = [("slices", f"{slice_size}")]
    return args


def populate_hslice_args(graph, nid, compiler_cfg):
    slices = graph["nodes"][nid]["forge_shape"][-3]

    args = [
        ("slices", f"{slices}"),
    ]
    return args


def populate_hstack_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    args = [
        ("slices", f"{input_shape[-3]}"),
    ]
    return args


def populate_index_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    strides = [int(strides) for strides in node["attrs"]["strides"][0]]
    begin = [int(begin) for begin in node["attrs"]["begin"][0]]
    end = [int(e) for e in node["attrs"]["end"][0]]

    assert len(strides) == 1 and len(begin) == 1 and len(end) == 1, "Strided slice should be on a single axis"
    assert int(node["attrs"]["num_inputs"]) == 1

    assert len(list(node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"
    dim = int(node["attrs"]["axes"][0][0])
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    # Use negative indexing
    if dim >= 0:
        dim -= len(input_shape)

    if end[0] == (2**31 - 1):  # Max int32 - To the end from TVM way of processing data
        end[0] = node["attrs"]["shape"][0][0][dim]

    args = []
    args.append(
        (
            "dim",
            f"{dim}",
        )
    )
    args.append(
        (
            "start",
            f"{begin[0]}",
        )
    )
    args.append(
        (
            "stop",
            f"{end[0]}",
        )
    )
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )
    return args


def populate_broadcast_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    output_shape = node["attrs"]["shape"][0][0]

    dim = 0
    shape = output_shape[dim]
    for i, (inp_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
        if inp_dim == out_dim:
            continue

        dim = i
        shape = out_dim
        input_shape[i] = out_dim
        assert input_shape == output_shape, "Forge broadcast only supports 1 dim"

    dim = dim - len(input_shape)
    args = []
    args.append(
        (
            "dim",
            f"{dim}",
        )
    )
    args.append(
        (
            "shape",
            f"{shape}",
        )
    )
    return args


def populate_reduce_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    dim = int(node["attrs"]["axis"][0][0])
    assert len(node["attrs"]["axis"][0]) == 1, "Forge only supports reduce with a single axis"

    keep_dim = int(node["attrs"]["keepdims"][0][0])

    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    if dim >= 0:
        dim -= len(input_shape)

    args = [("dim", f"{dim}"), ("keep_dim", f"{bool(keep_dim)}")]
    return args


def populate_select_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    begin = [int(x) for x in node["attrs"]["begin"][0]]
    end = [int(x) for x in node["attrs"]["end"][0]]
    strides = [int(x) for x in node["attrs"]["strides"][0]]

    assert end[0] > begin[0], "Select end must be bigger than begin"
    assert len(list(node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"
    dim = int(node["attrs"]["axes"][0][0])

    assert dim is not None
    if dim >= 0:
        dim -= len(input_shape)

    assert int(input_shape[dim]) % int(strides[0]) == 0, "Shape must be divisible by stride"
    select_stride = int(input_shape[dim]) // int(strides[0])
    args.append(
        (
            "dim",
            f"{dim}",
        )
    )
    args.append(
        (
            "index",
            f"({begin[0]}, {end[0]})",
        )
    )
    args.append(
        (
            "stride",
            f"{select_stride}",
        )
    )
    return args


def populate_softmax_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    dim = int(node["attrs"]["axis"][0][0])
    args = [("dim", f"{dim}")]
    return args


def populate_layernorm_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    epsilon = float(node["attrs"]["epsilon"][0][0])
    epsilon = round(epsilon, 10)

    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    dim = int(node["attrs"]["axis"][0][0])
    if dim > 0:
        dim = dim - len(input_shape)
    args = []
    args.append(("dim", f"{dim}"))
    args.append(("epsilon", f"{epsilon}"))

    return args


def populate_cast_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []
    dtype = node["attrs"]["dtype"][0][0]
    args.append(("dtype", pytorch_df_str_from_str(dtype, node["forge_name"])))
    return args


def populate_transpose_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    axes = [int(axis) for axis in node["attrs"]["axes"][0]]
    transpose_shape = list(graph["nodes"][nid]["forge_shape"])

    assert int(node["attrs"]["num_inputs"]) == 1

    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] += len(transpose_shape)

    node["attrs"]["axes"] = axes

    transpose_axes = []
    for idx, axis in enumerate(axes):
        if axis != idx:
            transpose_axes.insert(0, axis)

    # Tmp. Needs to be removed after full Jax Bert support
    if len(transpose_axes) == 0:
        transpose_axes = axes

    assert len(transpose_axes) == 2, "only single axis transpose supported at this time, decompose in tvm"

    transpose_axes = [axis - len(transpose_shape) for axis in transpose_axes]

    args = []
    args.append(("dim0", f"{transpose_axes[0]}"))
    args.append(("dim1", f"{transpose_axes[1]}"))

    return args


def populate_reshape_args(graph, nid, compiler_cfg):
    output_shape = graph["nodes"][nid]["forge_shape"]
    args = []
    args.append(("shape", f"{output_shape}"))
    return args


def populate_concatenate_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    forge_shape = node["forge_shape"]
    concat_axis = int(node["attrs"]["axis"][0][0])
    if concat_axis >= 0:
        concat_axis -= len(forge_shape)

    args = [
        ("axis", f"{concat_axis}"),
    ]
    return args


def populate_repeat_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    forge_shape = node["forge_shape"]
    reps = list(map(int, node["attrs"]["reps"][0]))
    args = [("repeats", f"{reps}")]
    return args


def populate_repeat_interleave_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    repeats = int(node["attrs"]["repeats"][0][0])
    dim = int(node["attrs"]["axis"][0][0])

    args = [("repeats", f"{repeats}"), ("dim", f"{dim}")]
    return args


def populate_stack_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    forge_shape = node["forge_shape"]
    stack_axis = int(node["attrs"]["axis"][0][0])
    if stack_axis >= 0:
        stack_axis -= len(forge_shape)

    args = [
        ("axis", f"{stack_axis}"),
    ]
    return args


def populate_clip_transpose_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    min = float(node["attrs"]["a_min"][0][0])
    max = float(node["attrs"]["a_max"][0][0])

    if min == float("inf"):
        min = "float('inf')"

    if max == float("inf"):
        max = "float('inf')"

    args.append(("min", f"{min}"))
    args.append(("max", f"{max}"))
    return args


def populate_pad_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    pad_width = [int(x) for x in node["attrs"]["pad_width"][0]]
    shape = node["attrs"]["shape"][0][0]
    channel_last = False

    mode = node["attrs"]["pad_mode"][0][0]
    assert mode in ["constant", "edge", "reflect"], "Forge pad only support constant/replicate/reflect padding for now"
    if len(shape) > 2:
        # Forge Pad only supports padding on last 2 dims
        assert len(pad_width) == len(shape) * 2
        assert all([x == 0 for x in pad_width[0:-6]]), "Forge Pad does not support padding on W dim"
        assert all([x == 0 for x in pad_width[-6:-4]]) or all(
            [x == 0 for x in pad_width[-2:]]
        ), "Forge only support Z dim padding for channel-last inputs"
        if any([x != 0 for x in pad_width[-6:-4]]):
            pad_width = pad_width[-6:-2]
            channel_last = True
        else:
            pad_width = pad_width[-4:]

    # TVM nn.pad axis start from the last axis, need to swap
    pad_width_by_axis = [pad_width[x : x + 2] for x in range(0, len(pad_width), 2)]
    pad_width_by_axis.reverse()
    pad_width_final = [item for axis in pad_width_by_axis for item in axis]

    if len(pad_width_final) == 2:
        args.append(
            (
                "pad",
                f"({pad_width_final[0]}, {pad_width_final[1]})",
            )
        )
    elif len(pad_width_final) == 4:
        args.append(
            (
                "pad",
                f"({pad_width_final[0]}, {pad_width_final[1]}, {pad_width_final[2]}, {pad_width_final[3]})",
            )
        )
    else:
        assert False

    tvm_pad_mode_to_forge_mode = {
        "constant": "constant",
        "edge": "replicate",
        "reflect": "reflect",
    }

    args.append(
        (
            "mode",
            f'"{tvm_pad_mode_to_forge_mode[mode]}"',
        )
    )
    args.append(
        (
            "channel_last",
            f"{channel_last}",
        )
    )

    return args


def populate_resize2d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    sizes = [int(x) for x in node["attrs"]["size"][0]]
    assert len(sizes) == 2
    method = node["attrs"]["method"][0][0]

    assert (
        method == "nearest_neighbor" or method == "linear" or method == "bilinear"
    ), "Only support nearest neighbor and linear for now"
    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    args.append(
        (
            "sizes",
            f"[{sizes[0]}, {sizes[1]}]",
        )
    )
    args.append(
        (
            "method",
            f'"{method}"',
        )
    )

    coordinate_transform_mode = node["attrs"]["coordinate_transformation_mode"][0][0]
    align_corners = "True" if coordinate_transform_mode == "align_corners" else "False"
    args.append(
        (
            "align_corners",
            f"{align_corners}",
        )
    )

    channel_last = int(node["attrs"]["layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_resize3d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    sizes = [int(x) for x in node["attrs"]["size"][0]]
    assert len(sizes) == 3
    method = node["attrs"]["method"][0][0]

    assert (
        method == "nearest_neighbor" or method == "linear" or method == "bilinear"
    ), "Only support nearest neighbor and linear for now"
    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    args.append(
        (
            "sizes",
            f"[{sizes[0]}, {sizes[1]}, {sizes[2]}]",
        )
    )
    args.append(
        (
            "method",
            f'"{method}"',
        )
    )

    coordinate_transform_mode = node["attrs"]["coordinate_transformation_mode"][0][0]
    align_corners = "True" if coordinate_transform_mode == "align_corners" else "False"
    args.append(
        (
            "align_corners",
            f"{align_corners}",
        )
    )

    channel_last = int(node["attrs"]["layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_index_copy_args(graph, nid, compiler_cfg):
    dim = graph["nodes"][nid]["attrs"]["axis"][0][0]
    args = []
    args.append(("dim", f"{dim}"))
    return args


def populate_unsupported_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    args = []
    op_type = node["name"]
    args.append(("op_type", f'"{op_type}"'))
    output_shape = node["forge_shape"]
    args.append(("output_shape", f"{output_shape}"))
    for k, v in node["attrs"].items():
        if k == "num_outputs" or k == "num_inputs" or k == "shape":
            continue
        while isinstance(v, list) and len(v) == 1:
            v = v[0]
        args.append((k, f'"{v}"'))
    return args


def populate_leaky_relu_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    alpha = node["attrs"]["alpha"][0][0]

    args.append(("alpha", alpha))

    return args


def populate_gelu_args(graph, nid, compiler_cfg):

    args = []
    node = graph["nodes"][nid]
    approx = node["attrs"]["approximate"][0][0]
    args.append(
        (
            "approximate",
            f'"{approx}"',
        )
    )

    return args


def populate_pixel_shuffle_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    upscale_factor = node["attrs"]["upscale_factor"][0][0]
    args.append(("upscale_factor", f"{upscale_factor}"))

    return args


def populate_prelu_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    axis = int(node["attrs"]["axis"][0][0])

    forge_shape = node["forge_shape"]
    if axis >= 0:
        axis -= len(forge_shape)
    args.append(
        (
            "axis",
            f'"{axis}"',
        )
    )


# def populate_dropout_args(graph, nid, training):
#     args = []
#     node = graph["nodes"][nid]

#     p = node["attrs"]["rate"][0][0]
#     args = (("p", p), ("training", str(training)))

#     return args


def populate_quantize_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    args.append(("out_dtype", "torch." + node["attrs"]["out_dtype"][0][0]))
    args.append(("axis", f"{int(node['attrs']['axis'][0][0])}"))
    return args


def populate_dequantize_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    args.append(("axis", f"{int(node['attrs']['axis'][0][0])}"))
    args.append(("out_dtype", "torch." + node["attrs"]["dtype"][0][0]))

    return args


def populate_requantize_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    args.append(("axis", f"{int(node['attrs']['axis'][0][0])}"))
    args.append(("out_dtype", "torch." + node["attrs"]["out_dtype"][0][0]))

    return args


# keep sorted
tvm_to_forge_op_map = {
    "abs": "abs",
    "add": "add",
    "floor_mod": "remainder",
    "argmax": "argmax",
    "broadcast_to": "broadcast",
    "cast": "cast",
    "clip": "clip",
    "concatenate": "concatenate",
    "cos": "cos",
    "cumsum": "cumsum",
    "embedding": "embedding",
    "equal": "equal",
    "exp": "exp",
    "gelu": "gelu",
    "greater_equal": "greater_equal",
    "greater": "greater",
    "identity": "identity",
    "image.resize2d": "resize2d",
    "image.resize3d": "resize3d",
    "layernorm": "layernorm",
    "less_equal": "less_equal",
    "less": "less",
    "log": "log",
    "logical_and": "logical_and",
    "logical_not": "logical_not",
    "max": "reduce_max",
    "maximum": "maximum",
    "mean": "reduce_avg",
    "minimum": "minimum",
    "multiply": "multiply",
    "nn.avg_pool1d": "avg_pool1d",
    "nn.avg_pool2d": "avg_pool2d",
    "nn.avg_pool3d": "avg_pool3d",
    "nn.batch_matmul": "matmul",
    "nn.conv2d_transpose": "conv2d_transpose",
    "nn.conv2d": "conv2d",
    "nn.conv3d": "conv3d",
    "nn.leaky_relu": "leaky_relu",
    "nn.log_softmax": "log_softmax",
    "nn.matmul": "matmul",
    "nn.max_pool1d": "max_pool1d",
    "nn.max_pool2d": "max_pool2d",
    "nn.max_pool3d": "max_pool3d",
    "nn.pad": "pad",
    "nn.relu": "relu",
    "nn.softmax": "softmax",
    "not_equal": "not_equal",
    "pixel_shuffle": "pixel_shuffle",
    "power": "power",
    "nn.prelu": "prelu",
    "forge.adv_index": "adv_index",
    "forge.binary_stack": "binary_stack",
    "forge.forge_conv2d_transpose_with_bias": "conv2d_transpose",
    "forge.forge_conv2d_with_bias": "conv2d",
    "forge.concatenate": "concatenate",
    "forge.dropout": "dropout",
    "forge.hslice": "hslice",
    "forge.hstack": "hstack",
    "forge.matmul": "matmul",
    "forge.vslice": "vslice",
    "forge.vstack": "vstack",
    "reciprocal": "reciprocal",
    "reshape": "reshape",
    "scatter": "index_copy",
    "sigmoid": "sigmoid",
    "sigmoid": "sigmoid",
    "sin": "sin",
    "sqrt": "sqrt",
    "stack": "stack",
    "strided_slice": "index",
    "subtract": "subtract",
    "sum": "reduce_sum",
    "take": "take",
    "tanh": "tanh",
    "tile": "repeat",
    "repeat": "repeat_interleave",
    "transpose": "transpose",
    "where": "where",
    "expand_dims": "unsqueeze",
    "squeeze": "squeeze",
    # Quantization ops
    "qnn.quantize": "quantize",
    "qnn.dequantize": "dequantize",
    "qnn.requantize": "requantize",
    "qnn.dense": "matmul",
}

forge_op_to_function_name = {
    "abs": "forge.op.Abs",
    "add": "forge.op.Add",
    "remainder": "forge.op.Remainder",
    "adv_index": "forge.op.AdvIndex",
    "argmax": "forge.op.Argmax",
    "avg_pool1d": "forge.op.AvgPool1d",
    "avg_pool2d": "forge.op.AvgPool2d",
    "avg_pool3d": "forge.op.AvgPool3d",
    "binary_stack": "forge.op.BinaryStack",
    "broadcast": "forge.op.Broadcast",
    "cast": "forge.op.Cast",  # Datatype cast
    "clip": "forge.op.Clip",
    "concatenate": "forge.op.Concatenate",
    "conv2d_transpose": "forge.op.Conv2dTranspose",
    "conv2d": "forge.op.Conv2d",
    "conv3d": "forge.op.Conv3d",
    "cos": "forge.op.Cosine",
    "cumsum": "forge.op.CumSum",
    "dropout": "forge.op.Identity",  # (Temporary): change when forge supports dropout
    "embedding": "forge.op.Embedding",
    "equal": "forge.op.Equal",
    "exp": "forge.op.Exp",
    "gelu": "forge.op.Gelu",
    "greater_equal": "forge.op.GreaterEqual",
    "greater": "forge.op.Greater",
    "hslice": "forge.op.HSlice",
    "hstack": "forge.op.HStack",
    "identity": "forge.op.Identity",
    "index_copy": "forge.op.IndexCopy",
    "index": "forge.op.Index",
    "layernorm": "forge.op.Layernorm",
    "leaky_relu": "forge.op.LeakyRelu",
    "less_equal": "forge.op.LessEqual",
    "less": "forge.op.Less",
    "log_softmax": "forge.op.LogSoftmax",
    "log": "forge.op.Log",
    "logical_and": "forge.op.LogicalAnd",
    "logical_not": "forge.op.LogicalNot",
    "matmul": "forge.op.Matmul",
    "max_pool1d": "forge.op.MaxPool1d",
    "max_pool2d": "forge.op.MaxPool2d",
    "max_pool3d": "forge.op.MaxPool3d",
    "maximum": "forge.op.Max",
    "mean": "forge.op.ReduceAvg",
    "minimum": "forge.op.Min",
    "multiply": "forge.op.Multiply",
    "not_equal": "forge.op.NotEqual",
    "pad": "forge.op.Pad",
    "pixel_shuffle": "forge.op.PixelShuffle",
    "power": "forge.op.Power",
    "prelu": "forge.op.Prelu",
    "reciprocal": "forge.op.Reciprocal",
    "reduce_avg": "forge.op.ReduceAvg",
    "reduce_max": "forge.op.ReduceMax",
    "reduce_sum": "forge.op.ReduceSum",
    "relu": "forge.op.Relu",
    "repeat": "forge.op.Repeat",
    "repeat_interleave": "forge.op.RepeatInterleave",
    "reshape": "forge.op.Reshape",
    "resize2d": "forge.op.Resize2d",
    "resize3d": "forge.op.Resize3d",
    "select": "forge.op.Select",
    "sigmoid": "forge.op.Sigmoid",
    "sin": "forge.op.Sine",
    "softmax": "forge.op.Softmax",
    "sqrt": "forge.op.Sqrt",
    "stack": "forge.op.Stack",
    "subtract": "forge.op.Subtract",
    "take": "forge.op.AdvIndex",
    "tanh": "forge.op.Tanh",
    "transpose": "forge.op.Transpose",
    "unsupported": "Unsupported",
    "vslice": "forge.op.VSlice",
    "vstack": "forge.op.VStack",
    "where": "forge.op.Where",
    "unsqueeze": "forge.op.Unsqueeze",
    "squeeze": "forge.op.Squeeze",
    # Quantization ops
    "quantize": "forge.op.Quantize",
    "dequantize": "forge.op.Dequantize",
    "requantize": "forge.op.Requantize",
}
forge_ops_needing_arguments = {
    "argmax": populate_argmax_args,
    "avg_pool1d": populate_avgpool1d_args,
    "avg_pool2d": populate_avgpool2d_args,
    "avg_pool3d": populate_avgpool3d_args,
    "binary_stack": populate_binary_stack_args,
    "broadcast": populate_broadcast_args,
    "cast": populate_cast_args,
    "clip": populate_clip_transpose_args,
    "concatenate": populate_concatenate_args,
    "conv2d_transpose": populate_conv2d_transpose_args,
    "conv2d": populate_conv2d_args,
    "conv3d": populate_conv3d_args,
    "cumsum": populate_cumsum_args,
    "gelu": populate_gelu_args,
    "hslice": populate_hslice_args,
    "hstack": populate_hstack_args,
    "index_copy": populate_index_copy_args,
    "index": populate_index_args,
    "layernorm": populate_layernorm_args,
    "leaky_relu": populate_leaky_relu_args,
    "log_softmax": populate_softmax_args,
    "max_pool1d": populate_maxpool1d_args,
    "max_pool2d": populate_maxpool2d_args,
    "max_pool3d": populate_maxpool3d_args,
    "pad": populate_pad_args,
    "pixel_shuffle": populate_pixel_shuffle_args,
    "prelu": populate_prelu_args,
    "reduce_avg": populate_reduce_args,
    "reduce_max": populate_reduce_args,
    "reduce_sum": populate_reduce_args,
    "repeat": populate_repeat_args,
    "repeat_interleave": populate_repeat_interleave_args,
    "reshape": populate_reshape_args,
    "resize2d": populate_resize2d_args,
    "resize3d": populate_resize3d_args,
    "select": populate_select_args,
    "softmax": populate_softmax_args,
    "stack": populate_stack_args,
    "transpose": populate_transpose_args,
    "unsupported": populate_unsupported_args,
    "vslice": populate_vslice_args,
    "vstack": populate_vstack_args,
    "unsqueeze": populate_unsqueeze_args,
    "squeeze": populate_squeeze_args,
    # "dropout"                      : populate_dropout_args,
    # Quantization ops
    "quantize": populate_quantize_args,
    "dequantize": populate_dequantize_args,
    "requantize": populate_requantize_args,
}


class ModuleWrapper(ForgeModule):
    def __init__(self, torchmod, name):
        super().__init__(name=name)
        self.torchmod = torchmod

    def forward(self, *acts):
        return self.torchmod(*acts)


class NodeType(Enum):
    Activation = 1
    Parameter = 2
    Constant = 3


class Operation:
    """
    A class to store relevant code generation details about a specific operation.

    Attributes:
        function_name (str): The name of the function associated with the operation.
        node_name (str): The name of the node in the computation graph.
        output_name (str): The name of the output variable.
        input_names (list): A list of input variable names.
        input_shapes (list): A list of shapes corresponding to the input variables.
        input_dtypes (list): A list of dtypes corresponding to the input variables.
        args (list): A list of arguments for the operation.
        is_submodule_call (bool): A flag indicating if the operation is a submodule call (related to Torch 2.0).
        inputs_to_delete (list): A list of inputs to delete.
        loop_with (list): A list of loop variables.
        src_layer (optional): The source layer associated with the operation.
    """

    def __init__(
        self,
        function_name,
        output_name,
        node_name="",
        input_names=[],
        args=[],
        src_layer=None,
        input_shapes=[],
        input_dtypes=[],
        input_node_types=[],
    ):
        self.function_name = function_name
        self.node_name = node_name
        self.output_name = output_name
        self.input_names = input_names
        self.input_node_types = input_node_types
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.args = args
        self.is_submodule_call = False
        self.inputs_to_delete = []
        self.loop_with = []
        self.src_layer = src_layer


def get_framework(module):
    if isinstance(module, forge.module.PyTorchModule):
        framework = "pytorch"
    elif isinstance(module, forge.module.TFModule):  # or isinstance(module, tf.keras.layers.Layer):
        framework = "tensorflow"
    elif isinstance(module, forge.module.TFGraphDefModule):
        framework = "tf_graphdef"
    elif isinstance(module, forge.module.OnnxModule):
        framework = "onnx"
    elif isinstance(module, forge.module.MXNetModule):
        framework = "mxnet"
    elif isinstance(module, forge.module.JaxModule):
        framework = "jax"
    elif isinstance(module, forge.module.TFLiteModule):
        framework = "tflite"
    else:
        assert False, f"Unsupported framework: {type(module)}"

    return framework


counter = 0
generated_files = []


def cleanup_temporary_files():
    global generated_files
    for filename in generated_files:
        if os.path.exists(filename):
            os.remove(filename)

    generated_files = []


def get_forge_outputs(forge_mods, devices, forge_inputs):
    from forge.tensor import to_forge_tensors, to_pt_tensors

    for i, (mod, dev) in enumerate(zip(forge_mods, devices)):
        if dev == "CPUDevice":
            forge_inputs = to_pt_tensors(forge_inputs)
        else:
            forge_inputs = to_forge_tensors(to_pt_tensors(forge_inputs))
        forge_inputs = mod.forward(*forge_inputs)

    if not isinstance(forge_inputs, (list, tuple)):
        forge_inputs = [forge_inputs]
    return to_forge_tensors(forge_inputs)


def verify_framework_vs_forge_codegen(frame_outputs, forge_outputs, verify_cfg):
    from forge.op.eval import compare_tensor_to_golden

    test_pass = True
    for i, (golden, output) in enumerate(zip(frame_outputs, forge_outputs)):
        test_pass &= compare_tensor_to_golden(
            f"Framework vs. Forge codegen output {i}", golden, output.value(), is_forge=False, verify_cfg=verify_cfg
        )

        assert test_pass, f"Data mismatch on output {i} between framework and Forge codegen"
    logger.info("Verified python codegen agains framework")


def save_writers_metadata(modules, inputs, sorted_inputs, module_name):
    metadata = {}
    metadata["writers"] = []

    for index, module in enumerate(modules):
        metadata["writers"].append({})
        metadata["writers"][index]["module_directory"] = module.module_directory
        metadata["writers"][index]["class_name"] = module.class_name
        metadata["writers"][index]["dev"] = module.dev
        metadata["writers"][index]["module_name"] = module.module_name
        metadata["writers"][index]["filename"] = module.filename

    metadata["input_indices"] = {}
    for index, unsorted_input in enumerate(inputs):
        for sorted_index, sorted_input in enumerate(sorted_inputs):
            if sorted_input.dtype == unsorted_input.dtype and torch.equal(sorted_input, unsorted_input):
                metadata["input_indices"][index] = sorted_index

    metadata["framework"] = modules[0].framework

    filename = module_name + ".json"
    with open(os.path.join(modules[0].module_directory, filename), "w") as metadata_file:
        json.dump(metadata, metadata_file)
    return


def metadata_path(module_name):
    module_directory = "generated_modules"
    filename = module_name + ".json"
    return os.path.join(module_directory, filename)


def load_writers_metadata(module_name, inputs):
    filepath = metadata_path(module_name)
    assert os.path.exists(
        filepath
    ), f"{filepath} not found, has the test been run with FORGE_RELOAD_GENERATED_MODULES disabled and compiler_cfg.retain_tvm_python_files enabled"
    with open(filepath, "r") as metadata_file:
        metadata = json.load(metadata_file)

    input_indices = metadata["input_indices"]
    flattened_inputs, _, _ = flatten_inputs(inputs)
    ordered_inptus = [None] * len(flattened_inputs)

    for k, v in input_indices.items():
        ordered_inptus[int(k)] = flattened_inputs[v]

    module_writers = []
    for module in metadata["writers"]:
        writer = PythonWriter(module["module_name"], open_file=False)
        writer.module_directory = module["module_directory"]
        writer.class_name = module["class_name"]
        writer.dev = module["dev"]
        writer.module_name = module["module_name"]
        writer.filename = module["filename"]
        module_writers.append(writer)

    return module_writers, ordered_inptus


def generate_forge_module(
    framework_mod, inputs, compiler_cfg=None, graph_name=None, verify_cfg=None, clean_later=False, input_names=[]
):
    global counter

    if compiler_cfg is None:
        compiler_cfg = _get_global_compiler_config()

    if verify_cfg is None:
        verify_cfg = _get_global_verify_config()

    pytorch_inputs = to_pt_tensors(inputs)

    if graph_name is None:
        graph_name = framework_mod.name

    reload = bool(int(os.environ.get("FORGE_RELOAD_GENERATED_MODULES", "0")))
    if reload and not compiler_cfg.retain_tvm_python_files:
        compiler_cfg.retain_tvm_python_files = True
        if not os.path.exists(metadata_path(graph_name)):
            reload = False

    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        framework_outputs = framework_mod.cpu_eval_forward(*pytorch_inputs)

    if not reload:
        module_name = graph_name if counter == 0 else f"{graph_name}_{counter}"
        module_writers, flattened_inputs = compile_tvm_to_python(
            framework_mod,
            graph_name,
            pytorch_inputs,
            module_name=module_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
            input_names=input_names,
        )
    else:
        module_writers, flattened_inputs = load_writers_metadata(graph_name, inputs)

    counter += 1
    sys.path.append(".")

    forge_mods = []
    devices = []
    for writer in module_writers:
        module = importlib.import_module(writer.import_module_path())
        module = importlib.reload(module)

        TestClass = getattr(module, writer.class_name)

        devices.append(writer.dev)
        if writer.dev == "CPUDevice":
            forge_mod = forge.PyTorchModule(writer.module_name, TestClass())
            forge_mod.module.process_framework_parameters(framework_mod.module)
        else:
            forge_mod = TestClass(writer.module_name)

            if isinstance(framework_mod, forge.module.PyTorchModule) and (
                compiler_cfg.tvm_generate_op_tests or compiler_cfg.tvm_generate_unique_op_tests
            ):
                forge_mod.process_framework_parameters()
            else:
                forge_mod.process_framework_parameters(framework_mod.module)

            assert not any(
                [param.value() is None for param in forge_mod.get_parameters()]
            ), f"Could not retrieve parameters from framework and tvm"

        forge_mods.append(forge_mod)

        if not compiler_cfg.retain_tvm_python_files:
            global generated_files
            generated_files.append(writer.filename)
            param_filename = os.path.join(writer.module_directory, writer.module_name + "_params.pt")
            if os.path.exists(param_filename):
                generated_files.append(os.path.abspath(param_filename))

            if not clean_later:
                cleanup_temporary_files()

    if devices[0] == "CPUDevice":
        forge_inputs = forge.tensor.to_pt_tensors(flattened_inputs)
    else:
        forge_inputs = forge.tensor.to_forge_tensors(flattened_inputs)

    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        forge_outputs = get_forge_outputs(forge_mods, devices, forge_inputs)
        verify_framework_vs_forge_codegen(framework_outputs, forge_outputs, verify_cfg=verify_cfg)

    return forge_mods, devices, forge_inputs


def compile_tvm_to_python(
    framework_mod, graph_name, inputs, module_name=None, compiler_cfg=None, verify_cfg=None, input_names=[]
):
    if compiler_cfg is None:
        compiler_cfg = _get_global_compiler_config()

    is_training = False if verify_cfg == None else verify_cfg.test_kind.is_training()

    framework = get_framework(framework_mod)
    if framework == "pytorch":
        if is_training:
            framework_mod.module.train()
            verify_cfg.verify_tvm_compile = False
            logger.warning("Cannot verify TVM output vs. framework in training mode.")
        else:
            framework_mod.module.eval()

    # Path is needed for Onnx model verification against TVM compile.
    path = None
    if isinstance(framework_mod, OnnxModule):
        path = framework_mod.onnx_path
    elif isinstance(framework_mod, TFLiteModule):
        path = framework_mod.tflite_path

    # Load here to avoid importing tvm unnecessarily when this file is loaded
    from tvm.contrib.forge_compile import load_tvm_graph

    json_graphs, flattened_pytorch_inputs, weights = load_tvm_graph(
        inputs,
        framework_mod.module,
        compiler_cfg,
        graph_name,
        framework,
        path=path,
        verify_cfg=verify_cfg,
        input_names=input_names,
    )

    def _determine_node_dtype(node):
        if "framework_dtype" in node["attrs"].keys() and node["attrs"]["framework_dtype"] != "N/A":
            return node["attrs"]["framework_dtype"]
        else:
            logger.debug(
                f"Node '{node['forge_name']}' does not have a framework dtype specified. Using TVM generated dtype."
            )
            return node["attrs"]["dtype"][0][0]

    span_lexer = re.compile("\S+$").search

    def span_to_src_layer(node):
        if "span" not in node["attrs"]:
            return None
        match = span_lexer(node["attrs"]["span"])
        return match.group(0) if match is not None else None

    modules = []
    for graph_index, json_graph in enumerate(json_graphs):
        graph = json.loads(json_graph["graph"])

        is_cpu_pre = graph_index == 0 and json_graph["device"] == "cpu"

        output_nodes = [head[0] for head in graph["heads"]]
        # reshape nops are added in tvm to passthrough nodes, prune them
        def is_nop_reshape(nid):
            node = graph["nodes"][nid]
            if node["name"] != "reshape":
                return False

            input_shape = graph["nodes"][node["inputs"][0][0]]["attrs"]["shape"]
            node_shape = node["attrs"]["shape"]
            return input_shape == node_shape

        input_nodes = graph["arg_nodes"]

        graph_input_names = {}
        params = {}
        constants = {}
        ops = {}
        returns = {}
        returns_requiring_batch_dim_fix = []
        forge_inputs = [None] * len(flattened_pytorch_inputs)
        params_from_tvm = {}
        node_name_to_node_type = {}

        def make_parser_friendly_name(node, node_type):
            if framework == "tensorflow" or framework == "tf_graphdef" or framework == "jax":
                node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_")
            elif framework == "pytorch":
                node["forge_name"] = node["forge_name"].replace(".", "_")
            elif framework == "onnx":
                node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_").replace(":", "_")
            elif framework == "tflite":
                node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_").replace(":", "_")

            # Preventing variable names starting with an integer in generated python
            if node["forge_name"][0] in [f"{n}" for n in range(10)]:
                node["forge_name"] = f"{node_type}_{node['name']}"

        # Clean up Forge name
        for node in graph["nodes"]:
            node["forge_name"] = node["name"]

        # Check for unsupported ops
        has_unsupported_ops = False
        json_graph_op_map = tvm_to_forge_op_map if json_graph["device"] == "tt" else tvm_to_pytorch_op_map
        for nid, node in enumerate(graph["nodes"]):
            if node["op"] != "kernel":
                continue

            if node["name"] not in json_graph_op_map:
                has_unsupported_ops = True
                unsupported_msg = (
                    f"Encountered unsupported op node type: {node['name']}, on device: {json_graph['device']}"
                )
                logger.warning(unsupported_msg) if compiler_cfg.enable_tvm_unsupported_ops else logger.error(
                    unsupported_msg
                )
        if has_unsupported_ops:
            assert (
                compiler_cfg.enable_tvm_unsupported_ops
            ), "Encountered unsupported op types. Check error logs for more details"

        for nid, node in enumerate(graph["nodes"]):

            node["nid"] = nid
            node["users"] = []
            shape = node["attrs"]["shape"][0][0]
            node["forge_shape"] = tuple(shape)
            if node["op"] == "input":
                if node["name"] not in weights:
                    make_parser_friendly_name(node, "input_")
                    # TVM might not preserve input order; check json graph
                    inp_idx = nid
                    if "nid_to_input_idx" in json_graph.keys() and len(json_graph["nid_to_input_idx"]) != 0:
                        inp_idx = json_graph["nid_to_input_idx"][nid]
                        forge_inputs[inp_idx] = flattened_pytorch_inputs[inp_idx]

                    graph_input_names[inp_idx] = node["forge_name"]
                    node_name_to_node_type[node["forge_name"]] = NodeType.Activation
                    node["op"] = "*"
                    logger.trace(f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: input")
                else:
                    tensor, requires_grad = weights[node["name"]]
                    tensor.requires_grad = requires_grad

                    if (requires_grad or json_graph["device"] == "cpu") and len(tensor.shape) > 0:
                        # CPU PyTorch module requires non-trainable weights in a nn.Parameter with
                        # requires_grad=False, a constant buffer does not work.
                        params[node["nid"]] = (
                            node["forge_name"],
                            node["forge_shape"],
                            requires_grad,
                            _determine_node_dtype(node),
                        )
                        node["op"] = "parameter"
                        node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                        logger.trace(
                            f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: parameter, requires_grad: {requires_grad}"
                        )
                    else:
                        if torch.numel(tensor) == 1 and len(tensor.shape) == 0:
                            tensor = tensor.reshape((1,))
                        if len(tensor.shape) > 4 and all([x == 1 for x in tensor.shape[0:-4]]):
                            tensor = tensor.reshape(tensor.shape[-4:])
                        if requires_grad:
                            node["op"] = "parameter"
                            params[node["nid"]] = (
                                node["forge_name"],
                                tensor.shape,
                                requires_grad,
                                _determine_node_dtype(node),
                            )
                            node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                            logger.trace(
                                f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: Parameter"
                            )
                        else:
                            node["op"] = "constant"
                            constants[node["nid"]] = (node["forge_name"], tensor.shape, _determine_node_dtype(node))
                            node_name_to_node_type[node["forge_name"]] = NodeType.Constant
                            logger.trace(
                                f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: Constant"
                            )

            elif node["op"] == "const":
                if isinstance(json_graph["params"][node["name"]], np.ndarray):
                    tensor = torch.from_numpy(json_graph["params"][node["name"]])
                else:
                    tensor = torch.tensor(json_graph["params"][node["name"]])

                requires_grad = node["attrs"]["is_param"] != "0"

                if requires_grad and len(tensor.shape) > 0:
                    if tensor.dtype == torch.bool:
                        requires_grad = False
                        node["attrs"]["is_param"] = "0"
                    params_from_tvm[node["forge_name"]] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
                    params[node["nid"]] = (
                        node["forge_name"],
                        node["forge_shape"],
                        requires_grad,
                        _determine_node_dtype(node),
                    )
                    node["op"] = "parameter"
                    node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                    logger.trace(
                        f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: parameter, requires_grad: {requires_grad}"
                    )
                else:
                    if torch.numel(tensor) == 1 and len(tensor.shape) == 0:
                        tensor = tensor.reshape((1,))
                    if len(tensor.shape) > 4 and all([x == 1 for x in tensor.shape[0:-4]]):
                        tensor = tensor.reshape(tensor.shape[-4:])
                    params_from_tvm[node["forge_name"]] = tensor
                    node["op"] = "constant"
                    constants[node["nid"]] = (node["forge_name"], tensor.shape, _determine_node_dtype(node))
                    node_name_to_node_type[node["forge_name"]] = NodeType.Constant
                    logger.trace(f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: Constant")

            elif node["op"] == "kernel":
                op_map = tvm_to_forge_op_map if json_graph["device"] == "tt" else tvm_to_pytorch_op_map
                if node["name"] in op_map:
                    op_type = op_map[node["name"]]
                else:
                    op_type = "unsupported"
                node["op"] = op_type

                function_map = (
                    forge_op_to_function_name if json_graph["device"] == "tt" else pytorch_op_to_function_name
                )
                function_name = function_map[op_type]
                node["forge_name"] = op_type + f"_{nid}"

                args = ()
                argument_getter = (
                    forge_ops_needing_arguments if json_graph["device"] == "tt" else pytorch_ops_needing_arguments
                )
                if op_type in argument_getter:
                    if op_type == "dropout" and json_graph["device"] != "tt":
                        if is_training:
                            logger.warning(
                                "Dropout op cannot be cpu fallback in training mode due to the absence of rate/p(probability) argument and it may also result in pcc mismatch"
                            )
                        args = argument_getter[op_type](graph=graph, nid=nid, training=is_training)
                    else:
                        args = argument_getter[op_type](graph=graph, nid=nid, compiler_cfg=compiler_cfg)
                    assert args is not None

                if args == () and json_graph["device"] == "cpu" and op_type not in argument_getter:
                    _, args = _populate_torch_init_args(graph, nid)

                logger.trace(f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']}  type: op")

                assert "num_inputs" in node["attrs"]

                # TVM nn.pad has 2 inputs [Data, pad_value]
                # We need to assert pad_value being zero, then remove the constant
                if node["name"] == "nn.pad" and int(node["attrs"]["num_inputs"]) == 2:
                    pad_value_node = graph["nodes"][node["inputs"][1][0]]
                    pad_value_node_name = pad_value_node["name"]
                    pad_value = json_graph["params"][pad_value_node_name]
                    assert pad_value_node["nid"] in constants
                    assert not pad_value.any(), "Padding contains non-zero values"
                    del constants[pad_value_node["nid"]]
                    node["attrs"]["num_inputs"] = "1"

                if node["name"] == "qnn.quantize":
                    assert int(node["attrs"]["num_inputs"]) == 3
                    zp_node = graph["nodes"][node["inputs"][2][0]]
                    zp_node_name = zp_node["name"]
                    assert zp_node["nid"] in constants
                    zp_value = json_graph["params"][zp_node_name]
                    del constants[zp_node["nid"]]
                    args.append(("zero_point", f"{float(zp_value.item())}"))
                    node["attrs"]["num_inputs"] = "2"

                if node["name"] == "qnn.dequantize":
                    assert int(node["attrs"]["num_inputs"]) == 3

                    zp_node = graph["nodes"][node["inputs"][2][0]]
                    zp_node_name = zp_node["name"]
                    assert zp_node["nid"] in constants
                    zp_value = json_graph["params"][zp_node_name]
                    del constants[zp_node["nid"]]
                    args.append(("zero_point", f"{zp_value.item()}"))
                    node["attrs"]["num_inputs"] = "2"

                if node["name"] == "qnn.requantize":
                    assert int(node["attrs"]["num_inputs"]) == 5
                    inp_zp_node = graph["nodes"][node["inputs"][2][0]]
                    inp_zp_node_name = inp_zp_node["name"]
                    assert inp_zp_node["nid"] in constants
                    inp_zp_value = json_graph["params"][inp_zp_node_name]
                    args.append(("input_zero_point", f"{inp_zp_value.item()}"))

                    out_zp_node = graph["nodes"][node["inputs"][4][0]]
                    out_zp_node_name = out_zp_node["name"]
                    assert out_zp_node["nid"] in constants
                    out_zp_value = json_graph["params"][out_zp_node_name]
                    args.append(("output_zero_point", f"{out_zp_value.item()}"))

                    node["inputs"] = [node["inputs"][0], node["inputs"][1], node["inputs"][3]]
                    del constants[inp_zp_node["nid"]]
                    del constants[out_zp_node["nid"]]
                    node["attrs"]["num_inputs"] = "3"

                input_names = []
                input_shapes = []
                input_dtypes = []
                input_node_types = []
                for input_port in range(int(node["attrs"]["num_inputs"])):
                    input_nid = node["inputs"][input_port][0]
                    input_node = graph["nodes"][input_nid]
                    if "users" not in input_node:
                        input_node["users"] = []
                    input_node["users"].append(nid)
                    input_names.append(input_node["forge_name"])
                    input_shapes.append(input_node["forge_shape"])
                    input_dtypes.append(_determine_node_dtype(input_node))
                    if input_nid in params.keys():
                        input_node_types.append(NodeType.Parameter)
                    elif input_nid in constants.keys():
                        input_node_types.append(NodeType.Constant)
                    else:
                        input_node_types.append(NodeType.Activation)

                # Handle concatenate case when a single node name in referenced twice in the input list
                if node["name"] == "forge.concatenate" and len(input_names) == 1:
                    inp_shape = graph["nodes"][node["inputs"][input_port][0]]["attrs"]["shape"][0][0]
                    out_shape = node["attrs"]["shape"][0][0]

                    if inp_shape[:2] == out_shape[:2] and inp_shape[2] * 2 == out_shape[2]:
                        input_names = [input_names[0], input_names[0]]
                        input_shapes = [input_shapes[0], input_shapes[0]]
                        input_dtypes = [input_dtypes[0], input_dtypes[0]]
                        input_node_types = [input_node_types[0], input_node_types[0]]

                if json_graph["device"] == "tt" and node["name"] == "embedding":
                    input_names = [input_names[1], input_names[0]]
                    input_shapes = [input_shapes[1], input_shapes[0]]
                    input_dtypes = [input_dtypes[1], input_dtypes[0]]
                    input_node_types = [input_node_types[1], input_node_types[0]]

                node_name_to_node_type[node["forge_name"]] = NodeType.Activation
                ops[node["nid"]] = Operation(
                    function_name=function_name,
                    # node_name=node["forge_name"],
                    output_name=node["forge_name"],
                    input_names=input_names,
                    args=args,
                    src_layer=span_to_src_layer(node),
                    input_shapes=input_shapes,
                    input_dtypes=input_dtypes,
                    input_node_types=input_node_types,
                )

        if any([input is None for input in forge_inputs]):
            forge_inputs = flattened_pytorch_inputs

        for output_nid in output_nodes:
            output_node = graph["nodes"][output_nid]
            returns[output_nid] = output_node["forge_name"]
            if len(output_node["forge_shape"]) == 0:
                returns_requiring_batch_dim_fix.append(output_node["forge_name"])
            elif output_node["forge_shape"][0] != 1:
                returns_requiring_batch_dim_fix.append(output_node["forge_name"])

            new_output_nid = len(graph["nodes"])
            graph["nodes"].append(
                {
                    "forge_name": output_node["forge_name"] + "_output",
                    "op": "output",
                    "nid": new_output_nid,
                    "inputs": [[output_nid, 0, 0]],
                    "attrs": {"num_inputs": "1"},
                }
            )

        def replace_node_name(orig, new):
            for op in ops.values():
                while orig in op.input_names:
                    op.input_names[op.input_names.index(orig)] = new

            if orig in returns.values():
                returns[list(returns.keys())[list(returns.values()).index(orig)]] = new
                if orig in returns_requiring_batch_dim_fix:
                    returns_requiring_batch_dim_fix[returns_requiring_batch_dim_fix.index(orig)] = new

        submodule = False
        param_names = {}
        const_names = {}
        # if compiler_cfg.tvm_module_to_num_patterns.get(framework_mod.get_name(), None):
        #     match_subgraph_patterns = compiler_cfg.tvm_module_to_num_patterns[framework_mod.get_name()]

        #     ret = pypattern_matcher.lower_json_to_pattern_matcher(graph, match_subgraph_patterns)
        #     subgraph_matches = ret.subgraph_matches

        #     if len(subgraph_matches) > 1:
        #         submodule = True

        #         matched_params = {}
        #         matched_consts = {}
        #         matched_ops = {}
        #         submodule_input_ports = {}
        #         submodule_outputs = {}
        #         submodule_outputs_requiring_batch_dim_fix = []
        #         submodule_output_shapes = {}

        #         # Collect submodule IOs
        #         for orig_nid in subgraph_matches[0].keys():
        #             node = graph["nodes"][orig_nid]
        #             if "num_inputs" in node["attrs"]:
        #                 for input_port in range(int(node["attrs"]["num_inputs"])):
        #                     if node["inputs"][input_port][0] not in subgraph_matches[0]:
        #                         submodule_input_ports[orig_nid] = input_port

        #             node = graph["nodes"][orig_nid]
        #             if "users" in node:
        #                 for user in node["users"]:
        #                     if user not in subgraph_matches[0] and node["op"] != "*":
        #                         submodule_outputs[node["nid"]] = node["forge_name"]
        #                         if node["forge_shape"][0] != 1:
        #                             submodule_outputs_requiring_batch_dim_fix.append(node["forge_name"])

        #         # add ops for each submodule call
        #         idx = max(sorted(submodule_input_ports)) + 0.5
        #         input_nids = list(sorted(submodule_input_ports.keys()))

        #         input_nodes = [graph["nodes"][input_nid] if submodule_input_ports[input_nid] == -1 else graph["nodes"][graph["nodes"][input_nid]["inputs"][submodule_input_ports[input_nid]][0]] for input_nid in input_nids]
        #         submodule_inputs = {input_node["nid"]:input_node["forge_name"] for input_node in input_nodes}
        #         activations = [input_node_name for _, input_node_name in sorted(graph_input_names.items())]

        #         ops[idx] = Operation(
        #             function_name="self.layers[0]",
        #             output_name="layer_0",
        #             input_names=activations,
        #         )
        #         ops[idx].is_submodule_call = True

        #         output_nids = list(submodule_outputs.keys())
        #         assert len(output_nids) == 1, "TODO"

        #         for i in range(1, len(subgraph_matches)):
        #             #if the input node is in the submodule
        #             activations = []
        #             for input_nid in input_nids:
        #                 if submodule_input_ports[input_nid] == -1:
        #                     matched_nid = subgraph_matches[i][input_nid]
        #                 else:
        #                     matched_user = subgraph_matches[i][input_nid]
        #                     matched_nid = graph["nodes"][matched_user]["inputs"][submodule_input_ports[input_nid]][0]

        #                 idx = matched_nid + 0.5
        #                 activations.append(graph["nodes"][matched_nid]["forge_name"])

        #             # unlike ops, submodules should not have repeated inputs
        #             activations = list(dict.fromkeys(activations))
        #             ops[idx] = Operation(
        #                 function_name=f"self.layers[{i}]",
        #                 output_name=f"layer_{i}",
        #                 input_names=activations,
        #             )
        #             ops[idx].is_submodule_call = True

        #         # build submodule param / op dicts, remove from main
        #         for orig_nid in subgraph_matches[0].keys():
        #             if orig_nid in params:
        #                 matched_params[orig_nid] = params[orig_nid]
        #                 param_name = params[orig_nid][0]
        #                 param_names[param_name] = (f"layer_{0}", param_name)
        #                 del params[orig_nid]
        #                 for index, subgraph in enumerate(subgraph_matches[1:]):
        #                     param_names[params[subgraph[orig_nid]][0]] = (f"layer_{index + 1}", param_name)
        #                     del params[subgraph[orig_nid]]

        #             if orig_nid in constants:
        #                 matched_consts[orig_nid] = constants[orig_nid]
        #                 const_name = constants[orig_nid][0]
        #                 const_names[const_name] = (f"layer_{0}", const_name)
        #                 del constants[orig_nid]
        #                 for index, subgraph in enumerate(subgraph_matches[1:]):
        #                     const_names[constants[subgraph[orig_nid]][0]] = (f"layer_{index + 1}", const_name)
        #                     del constants[subgraph[orig_nid]]

        #             if orig_nid in ops:
        #                 matched_ops[orig_nid] = ops[orig_nid]
        #                 del ops[orig_nid]
        #                 for subgraph in subgraph_matches[1:]:
        #                     del ops[subgraph[orig_nid]]

        #         #replace references to outputs of each submodule with submodule
        #         for idx, subgraph in enumerate(subgraph_matches):
        #             name_to_replace = graph["nodes"][subgraph[output_nids[0]]]["forge_name"]

        #             replace_node_name(name_to_replace, f"layer_{idx}")

        # Some float types (e.g. tf.bfloat16) are not compatible with numpy
        # We must signal to the ForgeWriter if the model contains these types so it can implement the workaround
        contains_incompatible_np_floats = False
        if framework == "tensorflow":
            for weight in framework_mod.module.weights:
                if weight.dtype in ForgeWriter.incompatible_np_float_types:
                    contains_incompatible_np_floats = True

        current_module_name = module_name
        if current_module_name is None:
            current_module_name = graph_name

        if len(json_graphs) > 1:
            current_module_name += f"_{json_graph['device']}_{graph_index}"

        if json_graph["device"] == "tt":
            delete_inputs = not (
                (verify_cfg is not None and verify_cfg.verify_all) or compiler_cfg.enable_op_level_comparision
            )
            if not delete_inputs:
                logger.warning(
                    "Preserving Intermediate tensor values in ForgeModule forward may causes out-of-memory issues"
                )
            writer = ForgeWriter(
                current_module_name,
                framework,
                contains_incompatible_np_floats=contains_incompatible_np_floats,
                delete_inputs=delete_inputs,
            )
        else:
            writer = PyTorchWriter(current_module_name, source_framework=framework)

        writer.write_header()

        if submodule:
            writer.write_class_definition(matched_params, matched_consts, class_name="Submodel", is_submodel=True)
            if is_cpu_pre:
                writer.write_forward(
                    matched_ops, submodule_inputs, submodule_outputs, submodule_outputs_requiring_batch_dim_fix
                )
            else:
                writer.write_forward(matched_ops, submodule_inputs, submodule_outputs)
            writer.write_class_definition(params, constants, num_submodels=len(subgraph_matches))
        else:
            writer.write_class_definition(params, constants)

        # can submodules be called in a loop? IE one outputs into the next
        loop_start = None
        prev_op = None
        for key in sorted(ops):
            op = ops[key]
            if op.is_submodule_call:
                if prev_op is None:
                    loop_start = op
                    prev_op = op
                elif prev_op.output_name in ops[key].input_names:
                    if len(loop_start.loop_with) == 0:
                        input_index = ops[key].input_names.index(prev_op.output_name)
                        loop_start.output_name = loop_start.input_names[input_index]
                        start, end = re.search("\[\d\]", loop_start.function_name).span()
                        loop_start.loop_start_index = int(loop_start.function_name[start + 1 : end - 1])
                        loop_start.function_name = re.sub("\[\d\]", "[i]", loop_start.function_name)

                    loop_start.loop_with.append(ops[key])
                    prev_op = ops[key]
                    del ops[key]
                else:
                    loop_start = None
                    prev_op = None
            else:
                loop_start = None
                prev_op = None

        # if there are any unconsumed inputs in the framework graph, they will not be part of the
        # generated graph, so we should add dummy variables to cunsume them. This is only needed for
        # the first module.
        if graph_index == 0:
            for input_index, _ in enumerate(forge_inputs):
                if input_index not in graph_input_names:
                    graph_input_names[input_index] = f"unused_input_{input_index}"

        for key in sorted(ops):
            if len(ops[key].loop_with):
                replace_node_name(ops[key].loop_with[-1].output_name, ops[key].output_name)

        def delete_unneeded_outputs(ops, returns):
            consumers = {}
            op_outputs = set()
            for ret in returns.values():
                consumers[ret] = 1
            for op in ops.values():
                op_outputs.add(op.output_name)
                for input_name in op.input_names:
                    if input_name not in consumers:
                        consumers[input_name] = 1
                    else:
                        consumers[input_name] = consumers[input_name] + 1
            for key in sorted(ops):
                for input_name in ops[key].input_names:
                    if input_name not in op_outputs:
                        continue
                    assert input_name in consumers
                    consumers[input_name] = consumers[input_name] - 1
                    if consumers[input_name] == 0:
                        ops[key].inputs_to_delete.append(input_name)

        delete_unneeded_outputs(ops, returns)
        if is_cpu_pre:
            writer.write_forward(ops, graph_input_names, returns, returns_requiring_batch_dim_fix)
        else:
            writer.write_forward(ops, graph_input_names, returns)

        param_file_name = None
        if len(params_from_tvm):
            param_file_name = os.path.join(writer.module_directory, writer.module_name + "_params.pt")
            torch.save(params_from_tvm, param_file_name)

        if framework == "pytorch" and (compiler_cfg.tvm_generate_op_tests or compiler_cfg.tvm_generate_unique_op_tests):
            # Store named parameters
            named_params_file_name = os.path.join(writer.module_directory, writer.module_name + "_named_params.pt")
            named_parameters = dict(framework_mod.module.state_dict().items())
            torch.save(named_parameters, named_params_file_name)

            # Store named buffers
            named_buffers_file_name = os.path.join(writer.module_directory, writer.module_name + "_named_buffers.pt")
            named_buffers = dict(framework_mod.module.named_buffers())
            torch.save(named_buffers, named_buffers_file_name)

            # Generate Forge module parameter parser
            param_names.update(const_names)
            writer.write_param_parser(param_names, param_file_name, named_params_file_name, named_buffers_file_name)
        else:
            param_names.update(const_names)
            writer.write_param_parser(param_names, param_file_name)

        writer.close_file()

        modules.append(writer)

        if framework == "pytorch":

            # Generate single op tests based on requested model. Currently only supported
            # for PyTorch framework.
            if compiler_cfg.tvm_generate_op_tests:
                generate_op_tests(
                    ops,
                    current_module_name,
                    framework,
                    contains_incompatible_np_floats,
                    delete_inputs,
                    node_name_to_node_type,
                    params,
                    constants,
                    param_names,
                    param_file_name,
                    named_params_file_name,
                    named_buffers_file_name,
                )

            # Generate unique op tests based on requested model. Currently only supported
            # for PyTorch framework.
            elif compiler_cfg.tvm_generate_unique_op_tests:
                generate_unique_op_tests(
                    ops,
                    current_module_name,
                    framework,
                    contains_incompatible_np_floats,
                    delete_inputs,
                    node_name_to_node_type,
                    params,
                    constants,
                    param_names,
                    param_file_name,
                    named_params_file_name,
                    named_buffers_file_name,
                    compiler_cfg,
                )

            if compiler_cfg.tvm_generate_op_tests or compiler_cfg.tvm_generate_unique_op_tests:

                # Exit python progrems without error
                # - Two different exit methods depending on whether compile is run using
                # pytest, or as a standalone python script
                if "pytest" in sys.modules:
                    pytest.exit("Exiting test without error", returncode=0)
                else:
                    sys.exit(0)

    if compiler_cfg.retain_tvm_python_files:
        save_writers_metadata(modules, flattened_pytorch_inputs, forge_inputs, graph_name)

    return modules, forge_inputs


class OpArgs(dict):
    """
    OpArgs is dictionary subclass to store arguments in which argument name will be stored as dictionary key
    and argument values will be stored as dictionary values with additional utility methods for adding, removing,
    comparing and updating arguments.

    Methods:
        get_args_names: Returns a list of argument names.
        get_args_values: Returns a list of argument values.
        add_arg: Adds a new argument with a specified name and value.
        remove_arg: Removes an argument by name.
        update_arg: Updates the value of an existing argument by name.
        is_empty: Checks if the argument dictionary is empty.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_args_names(self):
        return list(self.keys())

    def get_args_values(self):
        return list(self.values())

    def add_arg(self, arg_name, arg_value):
        self[arg_name] = arg_value

    def remove_arg(self, arg_name):
        if arg_name in self:
            del self[arg_name]
        else:
            print(f"Arg '{arg_name}' not found.")

    def update_arg(self, arg_name, arg_value):
        if arg_name in self:
            self[arg_name] = arg_value
        else:
            print(f"Arg '{arg_name}' does not exist and cannot be updated.")

    def __eq__(self, other):
        if not isinstance(other, (OpArgs, dict)):
            return False

        for arg_name, arg_value in self.items():
            if arg_name in other.keys():
                if arg_value != other[arg_name]:
                    return False
            else:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_empty(self):
        return len(self) == 0

    def __str__(self):
        return f"Opargs({super().__str__()})"


class OperandsInfo:
    """
    Stores operands information which include operand types(i.e NodeType), shapes and dtypes
    Args:
        operand_types: List of Operand type(i.e NodeType)
        operand_shapes: List of Operand shape
                        For constant nodetype, instead of storing shape, will store constant tensor.
        operand_dtypes: List of Operand datatype

    Methods:
        get_operand_types: Returns the list of operand types.
        get_operand_shapes: Returns the list of operand shapes.
        get_operand_dtypes: Returns the list of operand data types.
    """

    def __init__(self, operand_types, operand_shapes, operand_dtypes):

        # If lengths of operand_types, operand_shapes, and operand_dtypes doesn't match, raises assertion error
        assert len(operand_types) == len(operand_shapes) and len(operand_shapes) == len(
            operand_dtypes
        ), "Operands Type, shape, datatypes are not equal"
        self.operand_types = operand_types
        self.operand_shapes = operand_shapes
        self.operand_dtypes = operand_dtypes

    def get_operand_types(self):
        return self.operand_types

    def get_operand_shapes(self):
        return self.operand_shapes

    def get_operand_dtypes(self):
        return self.operand_dtypes

    def __eq__(self, other):
        """
        Checks equality between two OperandsInfo objects by comparing types, shapes, and data types.

        Args:
            other (OperandsInfo): The other OperandsInfo object to compare with.

        Returns:
            bool: True if both objects have the same operand information, otherwise False.
        """
        if not isinstance(other, OperandsInfo):
            return False
        if (
            len(self.operand_types) != len(other.operand_types)
            or len(self.operand_shapes) != len(other.operand_shapes)
            or len(self.operand_dtypes) != len(other.operand_dtypes)
        ):
            return False
        for type1, type2 in zip(self.operand_types, other.operand_types):
            if type1 != type2:
                return False
        for shape1, shape2 in zip(self.operand_shapes, other.operand_shapes):
            # For constant nodetype, will get constant tensor, instead of shape.
            if isinstance(shape1, torch.Tensor) and isinstance(shape2, torch.Tensor):
                if not torch.equal(shape1, shape2):
                    return False
            else:
                if shape1 != shape2:
                    return False
        for dtype1, dtype2 in zip(self.operand_dtypes, other.operand_dtypes):
            if dtype1 != dtype2:
                return False
        return True

    def __str__(self):
        if len(self.operand_types) > 0 and len(self.operand_shapes) > 0 and len(self.operand_dtypes) > 0:
            operand_info = "["
            for operand_type, operand_shape, operand_dtype in zip(
                self.operand_types, self.operand_shapes, self.operand_dtypes
            ):
                if isinstance(operand_shape, torch.Tensor):
                    operand_info += f"Operand(type={operand_type}, shape=Tensor, dtype={operand_dtype}), "
                else:
                    operand_info += f"Operand(type={operand_type}, shape={operand_shape}, dtype={operand_dtype}), "
            operand_info += "]"
            return operand_info

        else:
            return "OperandsInfo is empty!"


class OpArgsOpNames:
    """
    Stores OpArgs and associated operand names.

    Initializes OpArgsOpNames with a given OpArgs and operand names.

    Args:
        args (OpArgs): The OpArgs object to associate with operand names.
        operand_names (list): List of operand names to associate with args.

    Data Members:
        opargs_opnames (list of tuples): Each tuple contains an OpArgs object and a list of operand names.
    """

    def __init__(self, args: OpArgs, operand_names: List[str]):
        self.opargs_opnames = [(args, [operand_names])]

    def get_opargs_opnames(self):
        return self.opargs_opnames

    def update(self, new_args, new_operand_names):
        """
        Append operand names if arguments match, otherwise adds new OpArgs and operand names.

        Args:
            new_args (OpArgs): New arguments to match against existing ones.
            new_operand_names (list): New operand names to associate if new_args matches.
        """
        args_matched = False
        for idx, (arg, opnames_list) in enumerate(self.opargs_opnames):
            if (arg.is_empty() and new_args.is_empty()) or arg == new_args:
                self.opargs_opnames[idx][1].append(new_operand_names)
                args_matched = True
                break
        if not args_matched:
            self.opargs_opnames.append((new_args, [new_operand_names]))

    def __str__(self):
        if len(self.opargs_opnames) > 0:
            uniqueoperation_info = ""
            for idx, (args, opnames_list) in enumerate(self.opargs_opnames, start=1):
                uniqueoperation_info += f"\t\t\t\t {idx})" + str(args) + "\n"
                for opnames_idx, opnames in enumerate(opnames_list):
                    uniqueoperation_info += f"\t\t\t\t\t\t {opnames_idx})" + str(opnames) + "\n"
            return uniqueoperation_info
        else:
            return "OpArgsOpNames is empty!"


class UniqueOperationInfo:
    """
    Stores operands and argument associated with operand names.

    Args:
        operands (OperandsInfo): Information about operand types, shapes, and dtypes.
        oparg_opnames (OpArgsOpNames): Argument associated with the operand names.

    Data Members:
        unique_operands_and_opargs_opnames (list of tuples): Each tuple contains an OperandsInfo object
                                                             and an OpArgsOpNames object.
    """

    def __init__(self, operands: OperandsInfo, oparg_opnames: OpArgsOpNames):
        self.unique_operands_and_opargs_opnames = [(operands, oparg_opnames)]

    def get_unique_operands_and_opargs_opnames(self):
        return self.unique_operands_and_opargs_opnames

    def add_operands_args(self, new_operands, new_args, new_operand_names):
        """
        Adds or updates operandsInfo and Opargs and operand names.

        Args:
            new_operands (OperandsInfo): Operands information.
            new_args (OpArgs): Operation arguments.
            new_operand_names (list): Operand names.
        """
        operands_matched = False
        for idx, (operands, oparg_opnames) in enumerate(self.unique_operands_and_opargs_opnames):
            if operands == new_operands:
                operands_matched = True
                self.unique_operands_and_opargs_opnames[idx][1].update(new_args, new_operand_names)
                break
        if not operands_matched:
            self.unique_operands_and_opargs_opnames.append((new_operands, OpArgsOpNames(new_args, new_operand_names)))

    def __str__(self):
        if len(self.unique_operands_and_opargs_opnames) > 0:
            uniqueoperation_info = ""
            for idx, (operands, oparg_opnames) in enumerate(self.unique_operands_and_opargs_opnames, start=1):
                uniqueoperation_info += f"\t\t {idx})" + str(operands) + "\n"
                uniqueoperation_info += str(oparg_opnames) + "\n"
            return uniqueoperation_info

        else:
            return "UniqueOperationInfo is empty!"


class UniqueOperations(dict):
    """
    UniqueOperations is dictionary subclass to store forge op function name as dictionary key and
    UniqueOperationInfo (i.e Unique operands and Op arguments associated with operand names) as dictionary values
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def validate_node_types(cls, operand_names, operand_types, node_name_to_node_type):
        """
        Validates that each operand type matches the corresponding node type.

        Args:
            operand_names (list): Names of operands.
            operand_types (list): Types of operands.
            node_name_to_node_type (dict): Mapping of operand names to node types.

        Returns:
            bool: True if validation passes, otherwise False.
        """
        for operand_name, operand_type in zip(operand_names, operand_types):
            if operand_type == NodeType.Parameter and node_name_to_node_type[operand_name] != NodeType.Parameter:
                return False
            if operand_type == NodeType.Constant and node_name_to_node_type[operand_name] != NodeType.Constant:
                return False
            if operand_type == NodeType.Activation and node_name_to_node_type[operand_name] != NodeType.Activation:
                return False
        return True

    @classmethod
    def create_unique_operations(
        cls, ops: Dict[int, Operation], node_name_to_node_type: Dict[str, NodeType], named_parameters
    ):
        """
        Creates unique operations by mapping operand and argument information to function names.

        Args:
            ops (dict): Dictionary of operation.
            node_name_to_node_type (dict): Mapping of node names to types.
            named_parameters (dict): Mapping of node name to model parameters and buffers.

        Returns:
            UniqueOperations: Populated UniqueOperations dictionary.
        """
        unique_operations = UniqueOperations()
        for nid in sorted(ops):
            forge_op_function_name = ops[nid].function_name
            operand_names = ops[nid].input_names
            operand_types = ops[nid].input_node_types
            assert UniqueOperations.validate_node_types(
                operand_names, operand_types, node_name_to_node_type
            ), "Operand node types is not matching with node_name_to_node_type"
            operand_shapes = ops[nid].input_shapes
            operand_dtypes = ops[nid].input_dtypes
            args = ops[nid].args
            assert (
                len(operand_types) == len(operand_names)
                and len(operand_names) == len(operand_shapes)
                and len(operand_shapes) == len(operand_dtypes)
            ), "Operands names, shape, dtypes are not equal"

            # Replace constant node operand shape with constant value for comparing with other constant value.
            operand_shapes = [
                named_parameters[operand_name] if operand_type == NodeType.Constant else operand_shape
                for operand_type, operand_shape, operand_name in zip(operand_types, operand_shapes, operand_names)
            ]
            new_operands = OperandsInfo(operand_types, operand_shapes, operand_dtypes)
            new_args = OpArgs(args)
            if forge_op_function_name in unique_operations.keys():
                unique_operations[forge_op_function_name].add_operands_args(new_operands, new_args, operand_names)
            else:
                unique_operations[forge_op_function_name] = UniqueOperationInfo(
                    new_operands, OpArgsOpNames(new_args, operand_names)
                )

        return unique_operations

    def __str__(self):
        if len(self) > 0:
            uniqueoperations_info = ""
            for forge_op_function_name, unique_operation in self.items():
                uniqueoperations_info += forge_op_function_name + ": \n"
                uniqueoperations_info += str(unique_operation) + "\n"
            return uniqueoperations_info
        else:
            return "UniqueOperations is empty!"


def generate_op_tests(
    ops,
    current_module_name,
    framework,
    contains_incompatible_np_floats,
    delete_inputs,
    node_name_to_node_type,
    params,
    constants,
    param_names,
    param_file_name,
    named_params_file_name,
    named_buffers_file_name,
):
    """
    Generates test modules for a list of operations.

    This function creates unique test modules for each operation in the provided list.
    It initializes a ForgeWriter to generate the necessary code for testing each operation,
    including headers, class definitions, forward functions, parameter parsers, and pytest functions.
    The generated tests are designed to run the operations as standalone tests.
    """
    for op_idx, nid in enumerate(sorted(ops)):

        # Create unique module name
        module_name = "test_" + current_module_name.lower() + str(op_idx)

        # Initialize Forge writer and generate header
        writer = ForgeWriter(
            module_name,
            framework,
            module_directory=f"generated_modules/single_ops/{current_module_name}",
            contains_incompatible_np_floats=contains_incompatible_np_floats,
            delete_inputs=delete_inputs,
        )
        writer.write_header(include_pytest_imports=True)

        # Generating test for a single op
        single_op = {nid: ops[nid]}

        forward_method_inputs = {}
        pytest_input_shapes_dtypes = []
        needed_params = {}
        needed_consts = {}

        # Check whether all the inputs for the single op are const and params.
        assert UniqueOperations.validate_node_types(
            single_op[nid].input_names, single_op[nid].input_node_types, node_name_to_node_type
        ), "Operand node type extracted is not matching with node_name_to_node_type"
        all_inputs_are_params_const = all(
            [
                True if (node_type == NodeType.Parameter or node_type == NodeType.Constant) else False
                for node_type in single_op[nid].input_node_types
            ]
        )

        # If all the inputs for the single op are parameter and constants, pass the parameter/constants as input activation to the forge module forward function,
        # otherwise, pass the input activation to the forge module forward function and set parameters/constant in Forge module constructor(__init__ function).
        if all_inputs_are_params_const:
            for idx, input_name in enumerate(single_op[nid].input_names):
                forward_method_inputs[idx] = input_name.replace(".", "_")
                single_op[nid].input_names[idx] = input_name.replace(".", "_")
                pytest_input_shapes_dtypes.append((input_name, single_op[nid].input_dtypes[idx]))
        else:
            for idx, (input_name, node_type) in enumerate(
                zip(single_op[nid].input_names, single_op[nid].input_node_types)
            ):
                if node_type in [NodeType.Parameter, NodeType.Constant]:
                    for param_nid, param in params.items():
                        param_name = param[0]
                        if param_name == input_name:
                            needed_params[param_nid] = param
                            break
                    for const_nid, constant in constants.items():
                        constant_name = constant[0]
                        if constant_name == input_name:
                            needed_consts[const_nid] = constant
                            break
                    continue

                if input_name not in forward_method_inputs.values():
                    forward_method_inputs[idx] = input_name
                    pytest_input_shapes_dtypes.append(
                        (single_op[nid].input_shapes[idx], single_op[nid].input_dtypes[idx])
                    )

        # Generate class definition for the parameters and constants that are need for the single forge op inputs in forward function.
        writer.write_class_definition(needed_params, needed_consts)

        # Forge module forward method output to be same as the op we're running
        forward_method_returns = {nid: single_op[nid].output_name}

        # Generate single op forge module forward function
        writer.write_forward(single_op, forward_method_inputs, forward_method_returns)
        need_process_framework_parameters_func = False
        if needed_params or needed_consts:
            need_process_framework_parameters_func = True
            writer.write_param_parser(param_names, param_file_name, named_params_file_name, named_buffers_file_name)

        need_model_parameter_function = any(
            [True if isinstance(shape, str) else False for shape, _ in pytest_input_shapes_dtypes]
        )
        if need_model_parameter_function:
            writer.write_model_parameter_function(param_file_name, named_params_file_name, named_buffers_file_name)

        # Generate pytest function that enables runing Forge Module as standalone test
        writer.write_pytest_function(
            [writer.class_name], framework, [pytest_input_shapes_dtypes], [need_process_framework_parameters_func]
        )
        writer.close_file()


def generate_unique_op_tests(
    ops,
    current_module_name,
    framework,
    contains_incompatible_np_floats,
    delete_inputs,
    node_name_to_node_type,
    params,
    constants,
    param_names,
    param_file_name,
    named_params_file_name,
    named_buffers_file_name,
    compiler_cfg,
):
    """
    Generates test modules for unique operation configurations.

    The function extracts unique operation configurations based on the operation names, operand types, shapes,
    and datatypes, as well as operation arguments (if any). For operation, a test module
    file is created, which includes a Forge module for different configurations and associated test cases.
    """

    # Load the named parameters, constants, and buffers from files
    named_parameters = torch.load(named_params_file_name)
    if param_file_name is not None:
        serialized_params = torch.load(param_file_name)
        named_parameters.update(serialized_params)
    named_buffers = torch.load(named_buffers_file_name)
    named_parameters.update(named_buffers)

    # Extract unique operations by comparing operands types, shapes and dtypes and arguments if any
    unique_operations = UniqueOperations.create_unique_operations(ops, node_name_to_node_type, named_parameters)

    logger.info(f"Unique Operations:\n{unique_operations}")

    def get_param_const(name):
        for nid, param in params.items():
            if param[0] == name:
                return nid, param
        for nid, const in constants.items():
            if const[0] == name:
                return nid, const
        logger.error(f"There is no paramter/constant with the name {name}")

    unique_operation_details = []
    for op_idx, forge_op_function_name in enumerate(sorted(unique_operations)):

        # Extract operation name from forge op function name
        op_name = forge_op_function_name.split(".")[-1].lower()

        module_name = "test_" + op_name

        # Initialize Forge writer and generate header with pytest specific imports
        writer = ForgeWriter(
            module_name,
            framework,
            module_directory=f"generated_modules/unique_ops/{current_module_name}",
            contains_incompatible_np_floats=contains_incompatible_np_floats,
            delete_inputs=delete_inputs,
        )
        writer.write_header(include_pytest_imports=True)

        # Get the unique operands and operation arguments assiocated the operand names
        unique_operands_and_opargs_opnames = unique_operations[
            forge_op_function_name
        ].get_unique_operands_and_opargs_opnames()

        pytest_input_shapes_and_dtypes_list = []
        forge_module_names = []
        process_framework_parameters_func_status_list = []
        module_idx = 0
        forge_module_list = []
        test_count = 0
        for operands_idx, (operands, opargs_opnames) in enumerate(unique_operands_and_opargs_opnames):

            for args_idx, (args, opnames_list) in enumerate(opargs_opnames.get_opargs_opnames()):

                operand_types = operands.get_operand_types()
                operand_shapes = operands.get_operand_shapes()
                operand_dtypes = operands.get_operand_dtypes()
                operand_names = opnames_list[0]

                # Check if all operands types are parameters or constants and change the operand type from
                # parameters or constants to activation and pass it as activation to forge module forward function
                all_params_const = all(
                    [
                        True if (operand_type == NodeType.Parameter or operand_type == NodeType.Constant) else False
                        for operand_type in operand_types
                    ]
                )
                if all_params_const:
                    operand_types = [NodeType.Activation] * len(operand_types)
                    operand_shapes = operand_names
                    operand_names = [op_name + "_input_" + str(idx) for idx in range(len(operand_names))]

                # Check if an existing Forge module matches the current operation configuration.
                # This involves comparing the number of inputs, operand types, activation operand count,
                # and arguments. If a match is found, further checks are made to ensure that the parameter
                # shapes and data types, or constants, match as well. If a match is found for either parameters
                # or constants, the new Forge module creation is skipped. If no match is found, a new Forge module
                # will be created for the current operation configuration.
                need_to_create_forge_module = True
                for forge_mod in forge_module_list:
                    if (
                        forge_mod["number_of_inputs"] == len(operand_types)
                        and forge_mod["operand_types"] == operand_types
                    ):
                        if (
                            forge_mod["number_of_activation"]
                            == len(
                                list(filter(lambda operand_type: operand_type == NodeType.Activation, operand_types))
                            )
                            and forge_mod["args"] == args
                        ):
                            param_shape_dtype_list = [
                                (operand_shape, operand_dtype) if operand_type == NodeType.Parameter else "None"
                                for operand_type, operand_shape, operand_dtype in zip(
                                    operand_types, operand_shapes, operand_dtypes
                                )
                            ]
                            const_list = [
                                operand_shape if operand_type == NodeType.Constant else "None"
                                for operand_type, operand_shape in zip(operand_types, operand_shapes)
                            ]
                            param_shape_dtype_list.remove("None")
                            const_list.remove("None")
                            if forge_mod["number_of_parameters"] > 0 and len(param_shape_dtype_list) > 0:
                                if len(param_shape_dtype_list) == forge_mod["number_of_parameters"]:
                                    params_shape_dtype_equal = all(
                                        [
                                            True if (shape1 == shape2 and dtype1 == dtype2) else False
                                            for (shape1, dtype1), (shape2, dtype2) in zip(
                                                forge_mod["param_shape_dtype_list"], param_shape_dtype_list
                                            )
                                        ]
                                    )
                                    if params_shape_dtype_equal:
                                        need_to_create_forge_module = False
                                        forge_module_names.append(forge_mod["class_name"])
                                        process_framework_parameters_func_status_list.append(
                                            forge_mod["process_framework_parameters_func"]
                                        )
                                        break
                            elif forge_mod["number_of_constants"] > 0 and len(const_list) > 0:
                                if len(const_list) == forge_mod["number_of_constants"]:
                                    const_equal = all(
                                        [
                                            True if torch.equal(const1, const2) else False
                                            for const1, const2 in zip(forge_mod["const_list"], const_list)
                                        ]
                                    )
                                    if const_equal:
                                        need_to_create_forge_module = False
                                        forge_module_names.append(forge_mod["class_name"])
                                        process_framework_parameters_func_status_list.append(
                                            forge_mod["process_framework_parameters_func"]
                                        )
                                        break
                            else:
                                need_to_create_forge_module = False
                                forge_module_names.append(forge_mod["class_name"])
                                process_framework_parameters_func_status_list.append(
                                    forge_mod["process_framework_parameters_func"]
                                )
                                break

                # If no matching Forge module was found, create a new one for the current operation configuration
                if need_to_create_forge_module:

                    # Generate class name and append it forge_module_names list for using it as pytest parameter.
                    class_name = current_module_name.lower() + op_name + str(module_idx)
                    class_name = class_name.title().replace("_", "")
                    forge_module_names.append(class_name)

                    needed_params = {}
                    needed_consts = {}
                    params_shape_dtype_list = []
                    const_list = []
                    forward_method_inputs = {}
                    new_operand_names = []

                    # Iterate through operand types and names to classify them as parameters, constants, or activations.
                    # Collect the necessary parameters and constants, and use them to generate the class definition and
                    # handle activations for the forward method inputs.
                    for idx, (operand_type, operand_name) in enumerate(zip(operand_types, operand_names)):
                        if operand_type == NodeType.Parameter:
                            nid, param_tuple = get_param_const(operand_name)
                            needed_params[nid] = param_tuple
                            params_shape_dtype_list.append([param_tuple[1], param_tuple[3]])
                            new_operand_names.append(operand_name)
                        elif operand_type == NodeType.Constant:
                            nid, const_tuple = get_param_const(operand_name)
                            needed_consts[nid] = const_tuple
                            const_list.append(named_parameters[operand_name])
                            new_operand_names.append(operand_name)
                        else:
                            if operand_name not in forward_method_inputs.values():
                                forward_method_inputs[idx] = operand_name
                            else:
                                forward_method_inputs[idx] = op_name + "_input_" + str(idx)
                                logger.warning(
                                    f"operand_name {operand_name} is already present in the forward_method_inputs {forward_method_inputs}"
                                )
                            new_operand_names.append(forward_method_inputs[idx])

                    # Generate the class definition with the collected parameters and constants.
                    writer.write_class_definition(params=needed_params, constants=needed_consts, class_name=class_name)

                    # Create a single operation with the function name, output name,
                    # input operand names, and arguments and use it for generating forward method
                    single_op = {
                        args_idx: Operation(
                            function_name=forge_op_function_name,
                            output_name=op_name + "_output_1",
                            input_names=new_operand_names,
                            args=tuple(args.items()),
                        )
                    }

                    forward_method_returns = {args_idx: single_op[args_idx].output_name}

                    # Generate forge module forward function
                    writer.write_forward(single_op, forward_method_inputs, forward_method_returns)

                    # If there are any parameters or constants, generate the parameter parser function.
                    process_framework_parameters_func = False
                    if len(needed_params) != 0 or len(needed_consts) != 0:
                        process_framework_parameters_func = True
                        writer.write_param_parser(
                            param_names, param_file_name, named_params_file_name, named_buffers_file_name
                        )

                    module_idx += 1
                    process_framework_parameters_func_status_list.append(process_framework_parameters_func)
                    forge_module_list.append(
                        {
                            "class_name": class_name,
                            "process_framework_parameters_func": process_framework_parameters_func,
                            "number_of_inputs": len(operand_types),
                            "operand_types": operand_types,
                            "number_of_activation": len(forward_method_inputs),
                            "number_of_parameters": len(needed_params),
                            "number_of_constants": len(needed_consts),
                            "param_shape_dtype_list": params_shape_dtype_list,
                            "const_list": const_list,
                            "args": args,
                        }
                    )

                # Collect activation input shapes and dtypes for using it in pytest parameter
                pytest_input_shapes_dtypes = []
                for operand_type, operand_shape, operand_dtype in zip(operand_types, operand_shapes, operand_dtypes):
                    if operand_type == NodeType.Activation:
                        pytest_input_shapes_dtypes.append((operand_shape, operand_dtype))
                pytest_input_shapes_and_dtypes_list.append(pytest_input_shapes_dtypes)

                if compiler_cfg.export_tvm_generated_unique_op_tests_details:
                    operation_info = {}
                    operands_info = []
                    for node_type, name, shape, dtype in zip(
                        operand_types, operand_names, operand_shapes, operand_dtypes
                    ):
                        name_or_shape_val = name if node_type == NodeType.Constant else shape
                        operands_info.append(
                            f"Operand(type={node_type.name}, name/shape={name_or_shape_val}, dtype={dtype})"
                        )
                    operation_info["Framework"] = framework
                    operation_info["Op"] = op_name
                    operation_info["Operands"] = "\n".join(operands_info)
                    if args.is_empty():
                        operation_info["Args"] = ""
                    else:
                        operation_info["Args"] = "\n".join(
                            [f"{arg_name} : {arg_value}" for arg_name, arg_value in args.items()]
                        )
                    operation_info["tests"] = (
                        writer.module_directory
                        + "/"
                        + writer.filename
                        + f"::test_module[forge_module_and_shapes_dtypes{test_count}]"
                    )
                    unique_operation_details.append(operation_info)
                    test_count += 1

        # If the parameter/constant is passed as activation, operand shape will be replaced with operand name
        # because instead of generating tensor from shape, use actual tensor from model parameters/buffers
        # and so generating function for loading the model parameters/buffers and saving it as named_parameter variable
        need_model_parameter_function = any(
            [
                True if isinstance(shape, str) else False
                for pytest_input_shapes_dtypes in pytest_input_shapes_and_dtypes_list
                for shape, _ in pytest_input_shapes_dtypes
            ]
        )
        if need_model_parameter_function:
            writer.write_model_parameter_function(param_file_name, named_params_file_name, named_buffers_file_name)

        # Generate pytest function for the operation with pytest parameter containing list of tuple
        # and each tuple constaints module name, tuple of operand shape/name and dtype
        writer.write_pytest_function(
            forge_module_names,
            framework,
            pytest_input_shapes_and_dtypes_list,
            process_framework_parameters_func_status_list,
        )

        writer.close_file()

        if compiler_cfg.export_tvm_generated_unique_op_tests_details:
            xlsx_file_title = current_module_name
            xlsx_file_headers = ["Framework", "Op", "Operands", "Args", "Testfile"]
            xlsx_file_rows = []
            for operation_info in unique_operation_details:
                xlsx_file_rows.append(list(operation_info.values()))

            export_tvm_generated_unique_op_tests_details_dir_path = os.getenv(
                "FORGE_EXPORT_TVM_GENERATED_UNIQUE_OP_TESTS_DETAILS_DIR_PATH", f"generated_modules/unique_ops/"
            )
            if not os.path.exists(
                os.path.join(export_tvm_generated_unique_op_tests_details_dir_path, current_module_name)
            ):
                os.makedirs(
                    os.path.join(export_tvm_generated_unique_op_tests_details_dir_path, current_module_name),
                    exist_ok=True,
                )

            export_tvm_generated_unique_op_tests_details_file_path = os.path.join(
                export_tvm_generated_unique_op_tests_details_dir_path,
                current_module_name,
                "tvm_generated_op_test_details.xlsx",
            )

            create_excel_file(
                title=xlsx_file_title,
                headers=xlsx_file_headers,
                rows=xlsx_file_rows,
                output_file_path=export_tvm_generated_unique_op_tests_details_file_path,
            )
