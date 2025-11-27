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
    GemmNode,
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
    ConvNode,
)
from .pooling import (
    MaxPoolNode,
    AveragePoolNode,
    GlobalAveragePoolNode,
)
from .normalization import (
    BatchNormalizationNode,
)
from .shape import (
    FlattenNode,
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
)
from .generic import GenericNode

__all__ = [
    # Arithmetic
    'AddNode',
    'SubNode',
    'MulNode',
    'DivNode',
    'MatMulNode',
    'GemmNode',
    # Convolution
    'ConvNode',
    # Activation
    'ReluNode',
    'SigmoidNode',
    'TanhNode',
    'SoftmaxNode',
    'LogSoftmaxNode',
    'LeakyReluNode',
    # Pooling
    'MaxPoolNode',
    'AveragePoolNode',
    'GlobalAveragePoolNode',
    # Normalization
    'BatchNormalizationNode',
    # Shape
    'FlattenNode',
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
    # Generic
    'GenericNode',
]

