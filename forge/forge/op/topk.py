# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Iterator
import torch

from ..tensor import Tensor
from ..op.common import ForgeOp as op
from ..tensor import pytorch_dtype_to_forge_dataformat


class TensorPair:
    def __init__(self, values: Tensor, indices: Tensor):
        self.values = values
        self.indices = indices

    def set_src_layer(self, layer: str) -> "TensorPair":
        self.values.set_src_layer(layer)
        self.indices.set_src_layer(layer)
        return self

    def __iter__(self) -> Iterator[Tensor]:
        yield self.values
        yield self.indices

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> Tensor:
        if idx == 0:
            return self.values
        if idx == 1:
            return self.indices
        raise IndexError("TensorPair index out of range")

    def __getattr__(self, name):
        # Delegate tensor-like attributes/methods to values to behave as a Tensor by default
        return getattr(self.values, name)

    def as_tuple(self) -> Tuple[Tensor, Tensor]:
        return (self.values, self.indices)


def TopK(name: str, tensor: Tensor, k: int, dim: int, largest: bool = True, sorted: bool = True) -> TensorPair:
    """
    Top-K selection along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    tensor: Tensor
        Input tensor

    k: int
        The number of top elements to select along the given dimension

    dim: int
        Dimension along which to select Top-K.

    largest: bool
        If True, returns the k largest elements (default). If False, returns the k smallest elements.

    sorted: bool
        If True (default), the returned values are sorted by magnitude.

    Returns
    -------
    TensorPair
        Pair of tensors (values, indices) wrapped in a TensorPair. Behaves like a 2-tuple and supports set_src_layer().
    """
    # Anchor op for tracing and shape propagation
    topk_op = op("topk", name, tensor, attrs=(dim,), k=k, dim=dim, largest=largest, sorted=sorted)

    # Populate operand_broadcast needed by compile pipeline
    in_shape = tensor.shape.get_pytorch_shape()
    _, operand_broadcast = topk_op.cpp_op_type.shape([in_shape])
    topk_op.operand_broadcast = operand_broadcast

    in_dtype = tensor.pt_data_format if tensor.has_value() else torch.float32

    if tensor.has_value():
        values_pt, indices_pt = torch.topk(tensor.value(), k=k, dim=dim, largest=largest, sorted=sorted)
    else:
        out_shape = list(in_shape)
        out_shape[dim] = k
        values_pt = torch.zeros(tuple(out_shape), dtype=in_dtype)
        indices_pt = torch.zeros(tuple(out_shape), dtype=torch.int64)

    val_df = pytorch_dtype_to_forge_dataformat(values_pt.dtype)
    idx_df = pytorch_dtype_to_forge_dataformat(indices_pt.dtype)

    # Create Forge tensors from trace and set values
    FTensor = Tensor  # alias
    values_tt = FTensor.create_from_trace(topk_op, tuple(values_pt.shape), val_df, src_output_index=0)
    values_tt.set_value(values_pt)

    indices_tt = FTensor.create_from_trace(topk_op, tuple(indices_pt.shape), idx_df, src_output_index=1)
    indices_tt.set_value(indices_pt)

    return TensorPair(values_tt, indices_tt)
