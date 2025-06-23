# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

from ..tensor import Tensor
from ..parameter import Parameter
from forge.op.eval.forge import get_f_forge_eval, get_f_forge_shape
from forge._C import DataFormat
from forge._C.graph import OpType as OldOpType
from forge._C.ops import Op as CppOp, OpType
import forge
from forge.forgeglobal import get_unique_node_id, tracing
from forge.tensor import pytorch_dtype_to_forge_dataformat
from loguru import logger


# ==================================
# Op strings to new op type mappping
# ==================================


op_type_mapping = {
    "abs": OpType.Abs,
    "adaptive_max_pool2d": OpType.AdaptiveMaxPool2d,
    "add": OpType.Add,
    "adv_index": OpType.AdvIndex,
    "argmax": OpType.Argmax,
    "atan": OpType.Atan,
    "avg_pool1d": OpType.AvgPool1d,
    "avg_pool2d": OpType.AvgPool2d,
    "avg_pool3d": OpType.AvgPool3d,
    "batchnorm": OpType.Batchnorm,
    "broadcast": OpType.Broadcast,
    "buffer": OpType.Buffer,
    "cast": OpType.Cast,
    "clip": OpType.Clip,
    "concatenate": OpType.Concatenate,
    "constant": OpType.Constant,
    "conv2d": OpType.Conv2d,
    "conv2d_depthwise_weights": OpType.Conv2dDepthwiseWeights,
    "conv2d_depthwise_weights_bw": OpType.Conv2dDepthwiseWeightsBw,
    "conv2d_grouped_weights": OpType.Conv2dGroupedWeights,
    "conv2d_grouped_weights_bw": OpType.Conv2dGroupedWeightsBw,
    "conv2d_prestride_act": OpType.Conv2dPrestrideAct,
    "conv2d_prestride_weights": OpType.Conv2dPrestrideWeights,
    "conv2d_transpose": OpType.Conv2dTranspose,
    "conv3d": OpType.Conv3d,
    "conv_sum": OpType.ConvSum,
    "cosine": OpType.Cosine,
    "cumsum": OpType.CumulativeSum,
    "dequantize": OpType.Dequantize,
    "depthwise": OpType.Depthwise,
    "divide": OpType.Divide,
    "dram_queue": OpType.DramQueue,
    "dropout": OpType.Dropout,
    "embedding": OpType.Embedding,
    "embedding_bw": OpType.EmbeddingBw,
    "equal": OpType.Equal,
    "erf": OpType.Erf,
    "ethernet_datacopy": OpType.EthernetDatacopy,
    "exp": OpType.Exp,
    "forge_dequantize": OpType.ForgeDequantize,
    "forge_pad": OpType.ForgePad,
    "forge_quantize": OpType.ForgeQuantize,
    "forge_requantize": OpType.ForgeRequantize,
    "forge_unpad": OpType.ForgeUnpad,
    "gather": OpType.Gather,
    "gelu": OpType.Gelu,
    "gelu_derivative": OpType.GeluDerivative,
    "greater": OpType.Greater,
    "greater_equal": OpType.GreaterEqual,
    "grouped_reduce_avg": OpType.GroupedReduceAvg,
    "heaviside": OpType.Heaviside,
    "hslice": OpType.Hslice,
    "hstack": OpType.Hstack,
    "index": OpType.Index,
    "index_copy": OpType.IndexCopy,
    "interleave": OpType.Interleave,
    "layernorm": OpType.Layernorm,
    "layernorm_bw": OpType.LayernormBw,
    "leaky_relu": OpType.LeakyRelu,
    "less": OpType.Less,
    "less_equal": OpType.LessEqual,
    "log": OpType.Log,
    "log_softmax": OpType.LogSoftmax,
    "logical_and": OpType.LogicalAnd,
    "logical_not": OpType.LogicalNot,
    "mask": OpType.Mask,
    "matmul": OpType.Matmul,
    "maximum": OpType.Maximum,
    "minimum": OpType.Minimum,
    "multiply": OpType.Multiply,
    "nop": OpType.Nop,
    "not_equal": OpType.NotEqual,
    "narrow": OpType.Narrow,
    "pad": OpType.Pad,
    "pad_tile": OpType.PadTile,
    "pixel_shuffle": OpType.PixelShuffle,
    "pow": OpType.Pow,
    "power": OpType.Power,
    "quantize": OpType.Quantize,
    "reciprocal": OpType.Reciprocal,
    "reduce_avg": OpType.ReduceAvg,
    "reduce_max": OpType.ReduceMax,
    "reduce_sum": OpType.ReduceSum,
    "relu": OpType.Relu,
    "remainder": OpType.Remainder,
    "repeat": OpType.Repeat,
    "repeat_interleave": OpType.RepeatInterleave,
    "requantize": OpType.Requantize,
    "reshape": OpType.Reshape,
    "resize1d": OpType.Resize1d,
    "resize2d": OpType.Resize2d,
    "resize3d": OpType.Resize3d,
    "select": OpType.Select,
    "sigmoid": OpType.Sigmoid,
    "sine": OpType.Sine,
    "softmax": OpType.Softmax,
    "softmax_bw": OpType.SoftmaxBw,
    "sparse_matmul": OpType.SparseMatmul,
    "sqrt": OpType.Sqrt,
    "stack": OpType.Stack,
    "subtract": OpType.Subtract,
    "squeeze": OpType.Squeeze,
    "tanh": OpType.Tanh,
    "tile_broadcast": OpType.TileBroadcast,
    "tilizer": OpType.Tilizer,
    "transpose": OpType.Transpose,
    "unsqueeze": OpType.Unsqueeze,
    "upsample2d": OpType.Upsample2d,
    "vslice": OpType.Vslice,
    "vstack": OpType.Vstack,
    "where": OpType.Where,
}


def create_cpp_op(op_type: str, attrs):
    """
    Creates cpp_op based on provided op_type string.
    If op has new cpp implementation, we will create that op directly, otherwise, we will create
    base cpp op class and assing it op type from new cpp implementation.
    """
    return CppOp(op_type_mapping[op_type], attrs)


deprecated_name_dict = {}
deprecated_op_id = 0


class ForgeOp:
    def __init__(
        self, op_type: str, name: str, *operands: Union[Tensor, Parameter], attrs: Tuple[int, ...] = (), **named_attrs
    ):
        """
        Create an op with given parameters.
        """
        self.op_type = op_type

        global deprecated_op_id, deprecated_name_dict
        if tracing():
            if name != "":
                self.name = name
            else:
                unique_id = get_unique_node_id()
                self.name = f"{op_type}_{unique_id}"
                if unique_id != deprecated_op_id:
                    deprecated_name_dict[f"{op_type}_{deprecated_op_id}"] = self.name
        deprecated_op_id += 1

        operands = tuple(
            forge.op.Constant("", constant=operand) if isinstance(operand, (int, float)) else operand
            for operand in operands
        )
        self.operands = operands
        self.attrs = attrs
        self.named_attrs = named_attrs
        self.cpp_op_type = OldOpType(self.op_type, self.attrs, self.named_attrs)
        self.cpp_op = create_cpp_op(op_type=op_type, attrs=named_attrs)

    def get_tensor(self, out_df=None) -> Tensor:
        """
        Generate result tensor of the right shape, and if value is set, value.
        """

        # get reference output shape
        shapes = [o.shape.get_pytorch_shape() for o in self.operands]
        shape, self.operand_broadcast = self.cpp_op.shape(shapes)

        # get reference output value
        values = [o.value() if isinstance(o, (Tensor, Parameter)) else o for o in self.operands]
        ref_output = self.cpp_op.eval(values)

        if out_df is not None:  # User provided output dataformat
            data_format = out_df
        else:  # Use dataformat from the reference implementation if available (e.g. torch)
            # NOTE: This might need to be changed once we introduce config where each op can have its own dataformat
            # regardless of the reference implementation (e.g. running convolution in lower precision)
            data_format = pytorch_dtype_to_forge_dataformat(ref_output.dtype)

        result = Tensor.create_from_trace(src_op=self, shape=shape, data_format=data_format)
        result.requires_grad = any([o.requires_grad for o in self.operands])
        result.set_value(ref_output)

        return result
