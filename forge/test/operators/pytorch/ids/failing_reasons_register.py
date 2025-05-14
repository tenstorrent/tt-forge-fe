# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Register of all detected failing reasons by operator

from test.operators.utils import FailingReasons


class FailingReasonsRegister:

    # List of failing reasons for each operator
    # Format: (operator, failing_reason)
    xfail = [
        ("add", FailingReasons.DATA_MISMATCH),
        ("clamp", FailingReasons.DATA_MISMATCH),
        ("concatenate", FailingReasons.ALLOCATION_CIRCULAR_BUFFER),
        ("conv2d", FailingReasons.ALLOCATION_CIRCULAR_BUFFER),
        ("conv2d", FailingReasons.ALLOCATION_FAILED),
        ("conv2d", FailingReasons.CONV2D_VALIDATE_ARGS),
        ("conv2d", FailingReasons.DATA_MISMATCH),
        ("conv2d", FailingReasons.NOT_IMPLEMENTED),
        ("conv2d", FailingReasons.UNSUPPORTED_DATA_FORMAT),
        ("conv2d", FailingReasons.UNSUPPORTED_SPECIAL_CASE),
        ("cumsum", FailingReasons.DATA_MISMATCH),
        ("div", FailingReasons.DATA_MISMATCH),
        ("div", FailingReasons.DTYPE_MISMATCH),
        ("div", FailingReasons.SPECIAL_VALUES),
        ("embedding", FailingReasons.COMPILATION_FAILED),
        ("embedding", FailingReasons.DATA_MISMATCH),
        ("exp", FailingReasons.DATA_MISMATCH),
        ("ge", FailingReasons.DATA_MISMATCH),
        ("ge", FailingReasons.DTYPE_MISMATCH),
        ("gt", FailingReasons.DATA_MISMATCH),
        ("gt", FailingReasons.DTYPE_MISMATCH),
        ("isnan", FailingReasons.DTYPE_MISMATCH),
        ("layer_norm", FailingReasons.DATA_MISMATCH),
        ("layer_norm", FailingReasons.INTERNAL_TVM_ERROR),
        ("layer_norm", FailingReasons.UNSUPPORTED_DIMENSION),
        ("linear", FailingReasons.DATA_MISMATCH),
        ("linear", FailingReasons.MICROBATCHING_UNSUPPORTED),
        ("log", FailingReasons.SPECIAL_VALUES),
        ("log1p", FailingReasons.DATA_MISMATCH),
        ("log1p", FailingReasons.DTYPE_MISMATCH),
        ("log1p", FailingReasons.SPECIAL_VALUES),
        ("lt", FailingReasons.DATA_MISMATCH),
        ("lt", FailingReasons.DTYPE_MISMATCH),
        ("matmul", FailingReasons.DATA_MISMATCH),
        ("matmul", FailingReasons.INTERNAL_TVM_ERROR),
        ("matmul", FailingReasons.TTNN_RUNTIME),
        ("matmul", FailingReasons.UNSUPPORTED_DATA_FORMAT),
        ("max", FailingReasons.BUGGY_SHAPE),
        ("max", FailingReasons.DATA_MISMATCH),
        ("max", FailingReasons.FORGE_RUNTIME),
        ("max", FailingReasons.SPECIAL_VALUES),
        ("maximum", FailingReasons.DATA_MISMATCH),
        ("mean", FailingReasons.DATA_MISMATCH),
        ("minimum", FailingReasons.DATA_MISMATCH),
        ("mul", FailingReasons.DATA_MISMATCH),
        ("ne", FailingReasons.DTYPE_MISMATCH),
        ("neg", FailingReasons.DTYPE_MISMATCH),
        ("pow", FailingReasons.DATA_MISMATCH),
        ("pow", FailingReasons.SPECIAL_VALUES),
        ("pow", FailingReasons.UNSUPPORTED_SPECIAL_CASE),
        ("reciprocal", FailingReasons.DATA_MISMATCH),
        ("repeat_interleave", FailingReasons.DATA_MISMATCH),
        ("repeat_interleave", FailingReasons.INFERENCE_FAILED),
        ("reshape", FailingReasons.INTERNAL_TVM_ERROR),
        ("rsqrt", FailingReasons.SPECIAL_VALUES),
        ("softmax", FailingReasons.DATA_MISMATCH),
        ("softmax", FailingReasons.UNSUPPORTED_DATA_FORMAT),
        ("sqrt", FailingReasons.SPECIAL_VALUES),
        ("squeeze", FailingReasons.COMPILATION_FAILED),
        ("squeeze", FailingReasons.INTERNAL_TVM_ERROR),
        ("squeeze", FailingReasons.TVM_RUNTIME),
        ("sub", FailingReasons.DATA_MISMATCH),
        ("sub", FailingReasons.DTYPE_MISMATCH),
        ("sum", FailingReasons.DATA_MISMATCH),
        ("tanh", FailingReasons.DATA_MISMATCH),
        ("transpose", FailingReasons.DATA_MISMATCH),
        ("transpose", FailingReasons.MLIR_RUNTIME),
    ]

    # List of skip reasons for each operator
    # Format: (operator, skip_reason, failing_reason)
    skip = []
