# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

from ..tensor import Tensor
from ..parameter import Parameter
from forge.op.eval.forge import get_f_forge_eval, get_f_forge_shape
from forge._C import DataFormat
from forge._C.graph import OpType
import forge
from forge.forgeglobal import get_unique_node_id, tracing
from forge.tensor import pytorch_dtype_to_forge_dataformat
from loguru import logger


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
        self.cpp_op_type = OpType(self.op_type, self.attrs, self.named_attrs)

    def get_tensor(self, out_df=None) -> Tensor:
        """
        Generate result tensor of the right shape, and if value is set, value.
        """

        # get reference output shape
        shapes = [o.shape.get_pytorch_shape() for o in self.operands]
        shape, self.operand_broadcast = self.cpp_op_type.shape(shapes)

        # get reference output value
        values = [o.value() if isinstance(o, (Tensor, Parameter)) else o for o in self.operands]
        ref_output = self.cpp_op_type.eval(values)

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
