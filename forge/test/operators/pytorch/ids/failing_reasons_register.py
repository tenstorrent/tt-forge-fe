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
        ("clamp", FailingReasons.DTYPE_MISMATCH),
        ("clamp", FailingReasons.FORGE_RUNTIME),
        ("clamp", FailingReasons.SPECIAL_VALUES),
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
        ("embedding", FailingReasons.INDEX_ERROR),
        ("embedding", FailingReasons.SPECIAL_VALUES),
        ("exp", FailingReasons.DATA_MISMATCH),
        ("exp", FailingReasons.SPECIAL_VALUES),
        ("ge", FailingReasons.DATA_MISMATCH),
        ("ge", FailingReasons.DTYPE_MISMATCH),
        ("gt", FailingReasons.DATA_MISMATCH),
        ("gt", FailingReasons.DTYPE_MISMATCH),
        ("isnan", FailingReasons.DTYPE_MISMATCH),
        ("layer_norm", FailingReasons.DATA_MISMATCH),
        ("layer_norm", FailingReasons.INTERNAL_TVM_ERROR),
        ("layer_norm", FailingReasons.UNSUPPORTED_DIMENSION),
        ("layer_norm", FailingReasons.DTYPE_MISMATCH),
        ("layer_norm", FailingReasons.NOT_SUPPORTED_IN_TORCH),
        ("linear", FailingReasons.DATA_MISMATCH),
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
        ("max", FailingReasons.FORGE_RUNTIME),
        ("maximum", FailingReasons.DATA_MISMATCH),
        ("mean", FailingReasons.COMPILATION_FAILED),
        ("minimum", FailingReasons.DATA_MISMATCH),
        ("mul", FailingReasons.DATA_MISMATCH),
        ("ne", FailingReasons.DTYPE_MISMATCH),
        ("neg", FailingReasons.DTYPE_MISMATCH),
        ("pow", FailingReasons.DATA_MISMATCH),
        ("pow", FailingReasons.SPECIAL_VALUES),
        ("pow", FailingReasons.UNSUPPORTED_SPECIAL_CASE),
        ("remainder", FailingReasons.DATA_MISMATCH),
        ("remainder", FailingReasons.SPECIAL_VALUES),
        ("repeat_interleave", FailingReasons.INFERENCE_FAILED),
        ("repeat_interleave", FailingReasons.DATA_MISMATCH),
        ("repeat_interleave", FailingReasons.COMPILATION_FAILED),
        ("repeat_interleave", FailingReasons.SPECIAL_VALUES),
        ("reshape", FailingReasons.COMPILATION_FAILED),
        ("reshape", FailingReasons.INTERNAL_TVM_ERROR),
        ("rsqrt", FailingReasons.SPECIAL_VALUES),
        ("unsqueeze", FailingReasons.COMPILATION_FAILED),
        ("softmax", FailingReasons.DATA_MISMATCH),
        ("softmax", FailingReasons.UNSUPPORTED_DATA_FORMAT),
        ("sqrt", FailingReasons.SPECIAL_VALUES),
        ("squeeze", FailingReasons.COMPILATION_FAILED),
        ("squeeze", FailingReasons.INTERNAL_TVM_ERROR),
        ("sub", FailingReasons.DATA_MISMATCH),
        ("sub", FailingReasons.DTYPE_MISMATCH),
        ("sum", FailingReasons.DATA_MISMATCH),
        ("sum", FailingReasons.SPECIAL_VALUES),
        ("sum", FailingReasons.NOT_IMPLEMENTED),
        ("tanh", FailingReasons.DATA_MISMATCH),
        ("transpose", FailingReasons.COMPILATION_FAILED),
        ("transpose", FailingReasons.DATA_MISMATCH),
        ("transpose", FailingReasons.MLIR_RUNTIME),
    ]

    # List of skip reasons for each operator
    # Format: (operator, skip_reason, failing_reason)
    skip = []
