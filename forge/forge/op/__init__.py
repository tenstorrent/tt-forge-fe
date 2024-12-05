# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Local Imports
import forge.op.loss

from .constant import Constant
from .convolution import Conv2d, Conv2dTranspose, Conv3d
from .dram_queue import DRAMQueue
from .eltwise_binary import (
    Add,
    BinaryStack,
    Divide,
    Equal,
    Greater,
    GreaterEqual,
    Heaviside,
    Less,
    LessEqual,
    LogicalAnd,
    Max,
    Min,
    Multiply,
    NotEqual,
    Power,
    Remainder,
    Subtract,
)
from .eltwise_nary import Concatenate, IndexCopy, Interleave, Stack, Where
from .eltwise_unary import (
    Abs,
    Argmax,
    Buffer,
    Cast,
    Clip,
    Cosine,
    CumSum,
    Dropout,
    Exp,
    Gelu,
    Identity,
    LeakyRelu,
    Log,
    LogicalNot,
    Pow,
    Reciprocal,
    Relu,
    Sigmoid,
    Sine,
    Sqrt,
    Tanh,
    Tilize,
)
from .embedding import Embedding
from .matmul import Matmul, SparseMatmul
from .nn import Batchnorm, Layernorm, LogSoftmax, MaxPool2dModule, Softmax
from .pooling import AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d
from .quantize import Dequantize, ForgeRequantize, Quantize, Requantize
from .reduce import GroupedReduceAvg, ReduceAvg, ReduceMax, ReduceSum
from .resize import Resize2d, Resize3d
from .tm import (
    AdvIndex,
    Broadcast,
    ForgePad,
    ForgeUnpad,
    Index,
    Narrow,
    Pad,
    PadTile,
    PixelShuffle,
    Repeat,
    RepeatInterleave,
    Reshape,
    Select,
    Squeeze,
    Transpose,
    Unsqueeze,
)
