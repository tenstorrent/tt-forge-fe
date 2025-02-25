# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from tvm._ffi.base import TVMError
from tvm.ir.transform import PassContext
from tvm.relay import transform, analysis
from tvm.relay.dataflow_pattern import wildcard, is_op

import numpy as np
import math
import numpy as np
from tvm.relay.dataflow_pattern import *
import os

from loguru import logger



def fskip_eliminate(expr):
    if isinstance(expr, relay.expr.Call) and (expr.op.name == "transpose" or expr.op.name == "qnn.dequantize" or expr.op.name == "qnn.quantize"):
        return True
    return False

def run_relay_compile_passes(relay_module, print_all=False):

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    dump_flops = bool(int(os.environ.get("FORGE_SHOW_FLOPS_ESTIMATE", "0")))
    if dump_flops:
        total_macs = analysis.get_total_mac_number(relay_module["main"])
        total_gflops = (total_macs * 2) / 1e9
        logger.info(f"Initial flops estimate from TVM: {total_gflops}B")

    relay_module = tvm.transform.Sequential([transform.RemoveUnusedFunctions()])(relay_module)
    logger.trace("After RemoveUnusedFunctions")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.ToBasicBlockNormalForm()])(relay_module)
    logger.trace("After ToBasicBlockNormalForm")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.Legalize()])(relay_module)
    logger.trace("After Legalize")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.SimplifyInference()])(relay_module)
    logger.trace("After SimplifyInference")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.DynamicToStatic()])(relay_module)
    logger.trace("After DynamicToStatic")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.EliminateCommonSubexpr(fskip_eliminate)])(relay_module)
    logger.trace("After EliminateCommonSubexpr")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.SimplifyExpr()])(relay_module)
    logger.trace("After SimplifyExpr")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CombineParallelConv2D(3)])(relay_module)
    logger.trace("After CombineParallelConv2D")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CombineParallelBatchMatmul(3)])(relay_module)
    logger.trace("After CombineParallelBatchMatmul")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
    logger.trace("After FoldConstant")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CanonicalizeCast()])(relay_module)
    logger.trace("After CanonicalizeCast")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CanonicalizeOps()])(relay_module)
    logger.trace("After CanonicalizeOps")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
    logger.trace("After FoldConstant")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.Inline()])(relay_module)
    logger.trace("After Inline")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
    logger.trace("After FoldConstant")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CanonicalizeOps()])(relay_module)
    logger.trace("After CanonicalizeOps")
    logger.trace(relay_module.functions)

    # relay_module = tvm.transform.Sequential([relay.qnn.transform.CanonicalizeOps()])(relay_module)
    # logger.trace("After QNN CanonicalizeOps")
    # logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    return relay_module