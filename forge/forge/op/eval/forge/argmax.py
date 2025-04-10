# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional
from ..interface import PyEltwiseUnaryOp
import numpy as np


class Argmax(PyEltwiseUnaryOp):
    @classmethod
    def create(cls, dim=None, keep_dim=False):
        self = cls("argmax")
        self.dim = dim
        self.keep_dim = keep_dim

        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Argmax should have one input"

        # if dim is bool it is due to the way c++ pybind11 handles None
        if not hasattr(self, "dim") or isinstance(self.dim, bool):
            dim = None
        else:
            dim = self.dim

        ret = torch.argmax(tensors[0], dim=dim, keepdim=self.keep_dim)

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Argmax should have one input"

        # if dim is bool it is due to the way c++ pybind11 handles None
        if not hasattr(self, "dim") or isinstance(self.dim, bool):
            dim = None
        else:
            dim = self.dim

        input_shape = tensor_shapes[0]

        # Dimension-specific argmax
        if dim is not None:
            if self.keep_dim:
                shape = list(input_shape)
                shape[dim] = 1
            else:
                shape = [d for i, d in enumerate(input_shape) if i != dim]
        else:  # Global argmax across all dimensions
            if self.keep_dim:
                shape = [1] * len(input_shape)  # All dimensions become 1
            else:
                raise RuntimeError("This argmax reduction should return a scalar, but that's not supported yet")
                # shape = [] is what we should return
                # but we get this error RuntimeError: Unable to cast Python instance to C++ type

        return tuple(shape), []

    def backward(self, ac, operand, inputs, output, grad):
        raise RuntimeError("Argmax does not require grad and does not have a backwards function")

    def lower(self, lc, tensors, outputs):
        return None

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
