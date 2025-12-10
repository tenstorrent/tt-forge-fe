"""
Operations package for TIR nodes.
Imports all operations to register them.
"""
# Import all operations to register them
from .arithmetic import (
    AddNode,
    SubNode,
    MulNode,
    DivNode,
    MatMulNode,
)
from .activation import (
    ReluNode,
    SigmoidNode,
    TanhNode,
    SoftmaxNode,
    LogSoftmaxNode,
    LeakyReluNode,
)
from .conv import (
    Conv1dNode,
    Conv2dNode,
    Conv3dNode
)
from .pooling import (
    MaxPool1dNode,
    MaxPool2dNode,
    MaxPool3dNode,
    AveragePool1dNode,
    AveragePool2dNode,
    AveragePool3dNode,
    GlobalAveragePoolNode,
)
from .normalization import (
    BatchNormalizationNode,
)
from .shape import (
    ReshapeNode,
    TransposeNode,
    SqueezeNode,
    UnsqueezeNode,
)
from .reduction import (
    ReduceSumNode,
    ReduceMeanNode,
    ReduceMaxNode,
)
from .other import (
    ConcatNode,
    ClipNode,
    CastNode,
    IdentityNode,
    FullNode,
)

__all__ = [
    # Arithmetic
    'AddNode',
    'SubNode',
    'MulNode',
    'DivNode',
    'MatMulNode',
    # Convolution
    'Conv1dNode',
    'Conv2dNode',
    'Conv3dNode',
    # Activation
    'ReluNode',
    'SigmoidNode',
    'TanhNode',
    'SoftmaxNode',
    'LogSoftmaxNode',
    'LeakyReluNode',
    # Pooling
    'MaxPool1dNode',
    'MaxPool2dNode',
    'MaxPool3dNode',
    'AveragePool1dNode',
    'AveragePool2dNode',
    'AveragePool3dNode',
    'GlobalAveragePoolNode',
    # Normalization
    'BatchNormalizationNode',
    # Shape
    'ReshapeNode',
    'TransposeNode',
    'SqueezeNode',
    'UnsqueezeNode',
    # Reduction
    'ReduceSumNode',
    'ReduceMeanNode',
    'ReduceMaxNode',
    # Other
    'ConcatNode',
    'ClipNode',
    'CastNode',
    'IdentityNode',
    'FullNode',
]

