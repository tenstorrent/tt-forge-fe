# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from ..interface import PyTM
from ..lforge.transpose import TransposeTM as ForgeTransposeTM
from .. import sparse_utils
from forge._C import UnsupportedHWOpsError


class TransposeTM(PyTM):
    @classmethod
    def create(cls, dim0, dim1):
        self = cls("transpose")
        self.dim0 = dim0
        self.dim1 = dim1
        return self

    def eval(self, tensors):
        return torch.transpose(tensors[0], self.dim0, self.dim1)

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1
        shape = list(tensor_shapes[0])
        (shape[self.dim0], shape[self.dim1]) = shape[self.dim1], shape[self.dim0]
        return tuple(shape), []

    def backward(self, ac, operand, inputs, output, grad):
        assert operand == 0, "Invalid operand index"
        return ac.op(
            TransposeTM.create(self.dim0, self.dim1),
            [grad],
        )

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1

        if self.dim0 >= 0:
            self.dim0 -= tensors[0].shape.len()
        if self.dim1 >= 0:
            self.dim1 -= tensors[0].shape.len()

        if self.dim0 == -2 and self.dim1 == -1:
            lc.tm(
                ForgeTransposeTM.create(self.dim0, self.dim1),
                tensors[0],
            )
        else:
            raise UnsupportedHWOpsError(self)

    def decompose(self, dc, inputs):
        act = inputs[0]
        # canonicalize dims to use negative indexing
        if self.dim0 >= 0 or self.dim1 >= 0:
            if self.dim0 >= 0:
                self.dim0 -= inputs[0].shape.len()
            if self.dim1 >= 0:
                self.dim1 -= inputs[0].shape.len()
            dc.fuse(
                dc.op(
                    TransposeTM.create(self.dim0, self.dim1),
                    inputs,
                )
            )


def picker_matmul(use_sparse_mm, dc, s, result):
    if use_sparse_mm:
        lhs = dc.tensor(s)
        result = dc.op("sparse_matmul", [lhs, result])
    else:
        lhs = dc.tensor(s.to_dense())
        result = dc.op("matmul", [lhs, result])

    return result
