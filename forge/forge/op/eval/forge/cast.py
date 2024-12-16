# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import torch

from forge._C import DataFormat

from ....tensor import (
    forge_dataformat_to_pytorch_dtype,
    pytorch_dtype_to_forge_dataformat,
)
from ..interface import PyEltwiseUnaryOp


class Cast(PyEltwiseUnaryOp):
    @classmethod
    def create(cls, dtype: Union[torch.dtype, DataFormat]):
        self = cls("cast")
        self.dtype = pytorch_dtype_to_forge_dataformat(dtype).to_json()
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Cast should have one input"
        dtype = forge_dataformat_to_pytorch_dtype(DataFormat.from_json(self.dtype))
        ret = tensors[0].to(dtype)
        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Cast should have one input"
        shape = tensor_shapes[0]
        return shape, []
