# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Union, Tuple, List, Optional, Dict, TypeAlias
from forge.tvm_utils import map_pt_dtype_to_pd, map_tf_dtype_to_pt, map_pd_dtype_to_pt

import paddle
import torch
import tensorflow as tf
import numpy as np
import math
from loguru import logger
import copy
import jax
import jax.numpy as jnp
import json

from .forgeglobal import TILE_DIM, align_up_tile, round_up_div
from forge._C import DataFormat
from forge._C.graph import OpType, RuntimeTensorTransform, RuntimeTensorTransformType, get_constant_input_value
from forge.utils import detach_tensors
from functools import reduce
from operator import mul
from .utils import align_up

from forge.tvm_utils import map_tf_dtype_to_pt, map_pt_dtype_to_tf

import forge

SomeTensor = Union[torch.Tensor, "Tensor", np.ndarray]


class TensorShape:
    """
    Convenience wrapper for tensor dimensions. All Forge tensors are fixed to 4 dimensions - rows, columns, Z, W. Forge tensors are free to have any dimensions.
    """

    def __init__(self, *dims):
        self.dims = dims

    @property
    def r(self):
        """
        Alias for `rows`
        """
        assert len(self.dims) >= 2
        return self.dims[-2]

    @property
    def c(self):
        """
        Alias for `cols`
        """
        assert len(self.dims) >= 1
        return self.dims[-1]

    @property
    def rt(self):
        """
        Row tiles
        """
        return round_up_div(self.r, TILE_DIM)

    @property
    def w(self):
        """
        W dim
        """
        assert len(self.dims) >= 4
        return self.dims[-4]

    @property
    def z(self):
        """
        Z dim
        """
        assert len(self.dims) >= 3
        return self.dims[-3]

    @property
    def ct(self):
        """
        Column tiles
        """
        return round_up_div(self.c, TILE_DIM)

    def rc_divisible_by_tile_dim(self) -> bool:
        """
        Check if R/C dimensions are divisible by tile dimensions
        """
        return self.r % TILE_DIM == 0 and self.c % TILE_DIM == 0

    def get_pytorch_shape(self) -> Tuple[int, ...]:
        """
        Return this shape as a tuple used by pytorch/numpy - (w, z, r, c)
        """
        return self.dims

    @classmethod
    def from_pytorch_tuple(cls, w, z, r, c):
        """
        Create TensorShape from (w, z, r, c) tuple
        """
        return TensorShape(w, z, r, c)

    def numel(self):
        return len(self.dims)

    def __repr__(self):
        return str(self.get_pytorch_shape())

    def __len__(self):
        return self.numel()

    def __getitem__(self, i):
        return self.dims[i]

    def __eq__(self, other):
        return self.dims == other.dims


class TensorBase:
    def _create_const_tensor(self, value):
        assert isinstance(value, (int, float)), f"Automatic constant tensor creation for {type(value)} not supported"
        return forge.op.Constant("", constant=value)

    def _handle_binary_op(self, other, op, is_r=False):
        if not isinstance(other, (forge.Tensor, forge.Parameter)):
            other = self._create_const_tensor(other)
        if not is_r:
            return op("", self, other)
        else:
            return op("", other, self)

    def __add__(self, other):
        return self._handle_binary_op(other, forge.op.Add)

    def __radd__(self, other):
        return self._handle_binary_op(other, forge.op.Add, is_r=True)

    def __sub__(self, other):
        return self._handle_binary_op(other, forge.op.Subtract)

    def __rsub__(self, other):
        return self._handle_binary_op(other, forge.op.Subtract, is_r=True)

    def __mul__(self, other):
        return self._handle_binary_op(other, forge.op.Multiply)

    def __rmul__(self, other):
        return self._handle_binary_op(other, forge.op.Multiply, is_r=True)


class Tensor(TensorBase):
    """
    Common API for various Tensor versions - pytorch, traced tensor, tensor descriptor
    """

    def __init__(self):
        self.src_op = None
        self.src_layer = None

    def set_src_layer(self, layer: str):
        self.src_layer = layer
        return self

    def detach(self) -> "Tensor":
        return self  # by default, nothing to do. Children that have pytorch values will detach.

    def to_pytorch(self) -> torch.Tensor:
        return to_pt_tensors(self.value())[0]

    def to_tensorflow(self) -> tf.Tensor:
        return to_tf_tensors(self.value())[0]

    def to_jax(self) -> jnp.ndarray:
        return to_jax_tensors(self.value())[0]

    def to_framework(self, framework: str) -> "Tensor":
        """
        Convert to a tensor of a given framework
        """
        if not self.has_value():
            raise RuntimeError(f"Cannot convert to framework {framework} - tensor has no value")
        if framework == "pytorch":
            return self.to_pytorch()
        elif framework == "tensorflow":
            return self.to_tensorflow()
        elif framework == "jax":
            return self.to_jax()
        else:
            raise RuntimeError(f"Unknown framework {framework}")

    def has_value(self) -> bool:
        """
        Return true if Tensor has a concrete value
        """
        raise RuntimeError("Children should override")

    def value(self) -> torch.Tensor:
        """
        Returns the concrete pytorch tensor value
        """
        raise RuntimeError("Children should override")

    def to_forge_shape(self) -> "Tensor":
        """
        Returns a Tensor padded to 4d / tiles used by Forge.
        """
        raise RuntimeError("Children should override")

    # def to_tensor_desc(self) -> "PytorchTensorDesc":
    #     """
    #     Return a tensor descriptor, with shapes, strides, and a pointer to data buffer
    #     """
    #     raise RuntimeError("Children should override")

    def is_constant(self) -> bool:
        """
        Return whether or not the tensor is constant
        """
        return False

    def __repr__(self):
        if self.has_value():
            return f"Forge Tensor: {self.value()}, {self.data_format}"
        return f"Forge Empty Tensor: {self.shape}"

    def ndim(self) -> int:
        """
        Return tensor rank (number of dimensions)
        """
        return len(self.shape)

    @property
    def pt_data_format(self) -> torch.dtype:
        """
        Returns underlying pytorch tensor data format, if applicable
        """
        raise RuntimeError("Children should override")

    @property
    def data_format(self) -> DataFormat:
        """
        Returns data format used on the Tenstorrent device
        """
        raise RuntimeError("Children should override")

    def to_format(self, data_format: DataFormat) -> "Tensor":
        """
        Convert this tensor to data_format
        """
        raise RuntimeError("Children should override")

    def clone(self) -> "Tensor":
        """
        Create a copy of the tensor, along with any underlying torch tensors.
        """
        raise RuntimeError("Children should override")

    @property
    def shape(self):
        """
        Returns tensor shape
        """
        raise RuntimeError("Children should override")

    @classmethod
    def create_from_torch(
        cls, torch_tensor: torch.Tensor, dev_data_format: Optional[DataFormat] = None, constant: bool = False
    ) -> "TensorFromPytorch":
        """
        Create tensor from pytorch tensor, and set value
        """
        return TensorFromPytorch(torch_tensor, dev_data_format, constant)

    @classmethod
    def create_torch_tensor(
        cls,
        shape: Union[List[int], Tuple[int, ...], torch.Size],
        dtype: Optional[torch.dtype] = None,
        min_int: int = 0,
        max_int: int = 1000,
    ) -> torch.Tensor:

        if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            torch_tensor = torch.rand(shape, dtype=dtype)
        elif dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
            if min_int == max_int:
                torch_tensor = torch.full(size=shape, fill_value=max_int, dtype=dtype)
            else:
                torch_tensor = torch.randint(low=min_int, high=max_int, size=shape, dtype=dtype)
        elif dtype == torch.bool:
            torch_tensor = torch.randint(low=0, high=2, size=shape, dtype=dtype)  # this will create boolean tensor
        elif dtype is None:
            torch_tensor = torch.rand(shape, dtype=torch.float32)
        else:
            raise RuntimeError(f"[create_torch_tensor] - Unsupported dtype {dtype}")

        return torch_tensor

    @classmethod
    def create_from_shape(
        cls,
        tensor_shape: Union[List[int], Tuple[int, ...], torch.Size],
        torch_dtype: Optional[torch.dtype] = None,
        min_int: int = 0,
        max_int: int = 1000,
        constant: bool = False,
    ) -> "TensorFromPytorch":

        torch_tensor = Tensor.create_torch_tensor(
            shape=tensor_shape, dtype=torch_dtype, min_int=min_int, max_int=max_int
        )

        return TensorFromPytorch(
            torch_tensor, dev_data_format=pytorch_dtype_to_forge_dataformat(torch_dtype), constant=constant
        )

    @classmethod
    def create_from_trace(cls, src_op: "ForgeOp", shape: Tuple[int, ...], data_format: DataFormat) -> "TensorFromTrace":
        """
        New path to creating front-end Tensor
        """
        return TensorFromTrace(src_op, shape, data_format)

    # @classmethod
    # def create_from_tensor_descriptor(cls, descriptor: "PytorchTensorDesc") -> "TensorFromDescriptor":
    #     """
    #     New path to creating front-end Tensor
    #     """
    #     return TensorFromDescriptor(descriptor)


class TensorFromPytorch(Tensor):
    """
    Tensor wrapper created from a conrete pytorch tensor
    """

    def __init__(self, torch_tensor: torch.Tensor, dev_data_format: Optional[DataFormat], constant: bool):
        super().__init__()
        self._value = torch_tensor
        self._data_format: DataFormat = (
            dev_data_format if dev_data_format is not None else pytorch_dtype_to_forge_dataformat(self._value.dtype)
        )
        self._constant: bool = constant

    def has_value(self) -> bool:
        return True

    def value(self) -> torch.Tensor:
        return self._value

    def is_constant(self) -> bool:
        return self._constant

    @property
    def requires_grad(self) -> bool:
        return self._value.requires_grad

    def set_requires_grad(self, requires_grad: bool):
        self._value.requires_grad = requires_grad

    def to_forge_shape(
        self,
        tile_broadcast_dims: List[int],
        reinterpret_shape: Optional[List[int]] = None,
        clone: bool = False,
        squeeze: bool = False,
        microbatch=1,
        tile_r=TILE_DIM,
        tile_c=TILE_DIM,
    ) -> "Tensor":
        """
        Returns a Tensor padded to 4d / tiles used by Forge.
        """
        if clone:
            value = self._value.clone()
        else:
            value = self._value
            if is_forge_shape(value) and reinterpret_shape is None:
                return self

        if reinterpret_shape is not None:
            if reinterpret_shape[0] == 1:
                reinterpret_shape[0] = microbatch
            value = value.view(reinterpret_shape)
        new_tensor = pad_pytorch_tensor_to_forge(value, tile_broadcast_dims, squeeze, microbatch, tile_r, tile_c)
        return Tensor.create_from_torch(new_tensor)

    # def to_tensor_desc(self) -> "PytorchTensorDesc":
    #     """
    #     Creates a fully-populated descriptor if a pytorch tensor is set as value. Otherwise, an empty wrapper.
    #     """
    #     return pytorch_tensor_to_tensor_desc(self._value)

    @property
    def pt_data_format(self) -> torch.dtype:
        return self._value.dtype

    @property
    def data_format(self) -> DataFormat:
        return self._data_format

    def to_format(self, data_format: DataFormat) -> "Tensor":
        """
        Convert this tensor to data_format
        """
        new_pt_tensor = self._value.type(forge_dataformat_to_pytorch_dtype(data_format))
        return Tensor.create_from_torch(new_pt_tensor, dev_data_format=data_format)

    @property
    def shape(self):
        return TensorShape(*self._value.shape)

    def clone(self) -> Tensor:
        return Tensor.create_from_torch(self._value.clone())

    def detach(self) -> Tensor:
        from .utils import detach_tensors

        new_t = detach_tensors([self.value()])
        return Tensor.create_from_torch(new_t[0])

    def to_framework(self, framework: str) -> "Tensor":
        return super().to_framework(framework)


class TensorFromTrace(Tensor):
    """
    Tensor wrapper created by tracing model graph
    """

    def __init__(self, src_op: "ForgeOp", shape: Tuple[int, ...], data_format: DataFormat):
        super().__init__()
        self.tensor_shape = TensorShape(*shape)
        self.src_op = src_op
        self.requires_grad = False
        self._value = None
        self._data_format = data_format

    def has_value(self) -> bool:
        return self._value is not None

    def set_value(self, value: torch.Tensor):
        assert (
            self.tensor_shape.get_pytorch_shape() == value.shape
        ), f"Setting a tensor value of incorrect shape: {self.tensor_shape.get_pytorch_shape()} vs {value.shape}"

        self._value = value

    @property
    def shape(self):
        return self.tensor_shape

    def value(self) -> torch.Tensor:

        if self._value is not None:
            return self._value

        raise RuntimeError("Trying to get Tensor value where there isn't one")

    def clone(self) -> "TensorFromTrace":
        t: TensorFromTrace = Tensor.create_from_trace(
            self.src_op, self.tensor_shape.get_pytorch_shape(), self._data_format
        )
        t.requires_grad = self.requires_grad
        if self._value:
            t.set_value(self._value.clone())
        return t

    @property
    def pt_data_format(self) -> torch.dtype:
        if self._value is not None:
            return self._value.dtype

        raise RuntimeError("Trying to get Tensor value where there isn't one")

    @property
    def data_format(self) -> DataFormat:
        return self._data_format

    def to_format(self, data_format: DataFormat) -> "Tensor":
        """
        Convert this tensor to data_format
        """
        new_t = self.clone()
        new_t._data_format = data_format
        return new_t

    # def to_tensor_desc(self, batch: int = 0, override_data_format: DataFormat = DataFormat.Invalid) -> "PytorchTensorDesc":
    #     """
    #     Creates a descriptor, but doesn't assign a valid data pointer.
    #     Optionally modify the shape to add a batch value.

    #     Parameters
    #     ----------
    #     t: Tensor
    #         Forge tensor to be turned into a descriptor

    #     batch: int, optional
    #         If batch != 0, set batch dimension to given value
    #     """

    #     if self._value:
    #         return pytorch_tensor_to_tensor_desc(self._value)

    #     assert False

    def detach(self) -> Tensor:
        if self.has_value():
            from .utils import detach_tensors

            new_t = detach_tensors([self.value()])
            return Tensor.create_from_torch(new_t[0])

        return self  # nothing to detach

    def to_framework(self, framework: str) -> "Tensor":
        return super().to_framework(framework)


FrameworkTensor: TypeAlias = torch.Tensor | tf.Tensor | tf.Variable | paddle.Tensor | jax.Array
AnyTensor: TypeAlias = FrameworkTensor | Tensor


# class TensorFromDescriptor(Tensor):
#     """
#     Tensor wrapper created from tensor descriptor
#     """
#     def __init__(self, descriptor: "PytorchTensorDesc"):
#         super().__init__()
#         self.descriptor = descriptor
#         self.requires_grad = False

#     # Cloning a tensor from descriptor creates a pytorch tensor
#     def clone(self) -> "TensorFromTorch":
#         return Tensor.create_from_torch(self.value(clone=True))

#     def has_value(self) -> bool:
#         return True

#     def value(self, clone = False) -> torch.Tensor:
#         tensor = tensor_desc_to_pytorch_tensor(self.descriptor)
#         if clone:
#             return tensor.clone()

#         return tensor

#     def to_forge_shape(self) -> "Tensor":
#         raise RuntimeError("Tensor descriptor should not be converted to forge shape")

#     def to_tensor_desc(self) -> "PytorchTensorDesc":
#         return self.descriptor

#     # TODO: Can reinterpret shape be moved outside of this method?
#     def narrow_to_original_shape(self, original_shape: Tuple[int, ...], reinterpret_shape: Optional[Tuple[int, ...]] = None, has_microbatch_dim: bool = False, unpadded_shape: Optional[Tuple[int, ...]] = None) -> "Tensor":
#         """
#         Narrow the tensor to a smaller one, if original shape is smaller
#         """
#         assert type(original_shape) == tuple, "original_shape must be a tuple"

#         tensor = self.value()

#         if self.shape.get_pytorch_shape() == original_shape and (reinterpret_shape is None or len(reinterpret_shape) == 0):
#             return Tensor.create_from_torch(tensor)

#         shape_transform = original_shape if (reinterpret_shape is None or len(reinterpret_shape) == 0) else reinterpret_shape

#         new_shape = list(self.shape.get_pytorch_shape())
#         # Only R/C get narrowed
#         new_shape[-1] = shape_transform[-1]
#         if len(shape_transform) > 1:
#             new_shape[-2] = shape_transform[-2]
#             new_shape = (*shape_transform[:-2], new_shape[-2], new_shape[-1])
#             new_tensor = narrow_forge_tensor_to_pytorch(tensor, new_shape, has_microbatch_dim=has_microbatch_dim)
#         else:
#             new_shape = (new_shape[-1],)
#             new_tensor = narrow_forge_tensor_to_pytorch(tensor, new_shape, has_microbatch_dim=has_microbatch_dim)

#         new_tensor = new_tensor.reshape(original_shape)

#         # Reshape the rest
#         return Tensor.create_from_torch(new_tensor)

#     @property
#     def pt_data_format(self) -> torch.dtype:
#         return forge_dataformat_to_pytorch_dtype(self.descriptor.format)

#     @property
#     def data_format(self) -> DataFormat:
#         return self.descriptor.format

#     def to_format(self, data_format: DataFormat) -> "Tensor":
#         """
#         Convert this tensor to data_format
#         """
#         new_pt_tensor = self.value().type(forge_dataformat_to_pytorch_dtype(data_format))
#         new_pt_tensor.requires_grad = self.requires_grad
#         return Tensor.create_from_torch(new_pt_tensor, dev_data_format=data_format)

#     @property
#     def shape(self):
#         return TensorShape(*self.descriptor.shape)

#     def to_framework(self, framework: str) -> "Tensor":
#         return super().to_framework(framework)


# def verify_tile_dims(data, msg = "Dim check"):
#     """
#     Verify that data tensor, or all tensors in data list have rows and columns divisible with tile dimensions
#     """
#     if isinstance(data, (list, tuple)):
#         for d in data:
#             verify_tile_dims(d, msg)
#         return

#     if data.shape[-1] % TILE_DIM != 0:
#         raise RuntimeError(f"{msg}: Shape {data.shape}: Column of {data.shape[-1]} encountered, which is not divisible with tile dimension of {TILE_DIM}")

#     if data.shape[-2] % TILE_DIM != 0:
#         raise RuntimeError(f"{msg}: Shape {data.shape}: Row of {data.shape[-2]} encountered, which is not divisible with tile dimension of {TILE_DIM}")


def cast_unsupported_torch_dtype(tensor: torch.Tensor):
    """
    Casts a PyTorch tensor to the dtype that is supported in Forge.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor casted to the supported data format.
    """
    forge_dataformat = pytorch_dtype_to_forge_dataformat(tensor.dtype)

    # Get the corresponding PyTorch dtype
    new_dtype = forge_dataformat_to_pytorch_dtype(forge_dataformat)

    # If mapping exists and is different from the current dtype, cast the tensor
    if new_dtype and new_dtype != tensor.dtype:
        logger.warning(
            "Tensor dtype {} is not supported in forge. Therefore, it is casted in to {}", tensor.dtype, new_dtype
        )
        return tensor.to(new_dtype)

    # If no change is needed, return the original tensor
    return tensor


def pytorch_dtype_to_forge_dataformat(dtype: torch.dtype, fp32_fallback: Optional[DataFormat] = None) -> DataFormat:

    if isinstance(dtype, DataFormat):
        return dtype

    assert isinstance(dtype, torch.dtype)

    if dtype == torch.float16:
        return DataFormat.Float16

    if dtype == torch.bfloat16:
        return DataFormat.Float16_b

    if dtype == torch.float32:
        if fp32_fallback is not None:
            return fp32_fallback
        return DataFormat.Float32

    if dtype == torch.uint8:
        return DataFormat.RawUInt8

    if dtype == torch.int8:
        logger.warning("Parameter is int8. Setting to Int32, since int8 is not supported.")
        return DataFormat.Int32

    # These are kind of arbitrary..
    # if dtype == torch.uint8 or dtype == torch.int:
    #     return DataFormat.UInt16

    if dtype == torch.bool:
        logger.warning("Parameter is bool. Setting to uint8, since bool is not supported.")
        return DataFormat.RawUInt8

    if dtype == torch.int32:
        return DataFormat.Int32

    if dtype == torch.int64:
        logger.warning("Parameter is int64. Setting to int32, since int64 is not supported.")
        return DataFormat.Int32

    raise RuntimeError("Unsupported torch dtype " + str(dtype))


def forge_dataformat_to_pytorch_dtype(data_format: DataFormat) -> torch.dtype:
    assert isinstance(data_format, DataFormat)

    if data_format == DataFormat.Float32:
        return torch.float32

    if data_format == DataFormat.Float16:
        return torch.float16

    if data_format == DataFormat.Float16_b:
        return torch.bfloat16

    if data_format in [DataFormat.Bfp8_b, DataFormat.Bfp4_b]:
        return torch.bfloat16

    if data_format in [DataFormat.Bfp8, DataFormat.Bfp4, DataFormat.Bfp2]:
        return torch.float16

    if data_format == DataFormat.RawUInt8:
        return torch.uint8

    if data_format == DataFormat.RawUInt32:
        return torch.int

    if data_format == DataFormat.UInt16:
        return torch.int

    if data_format == DataFormat.Int32:
        return torch.int32

    raise RuntimeError("Unsupported DataFormat for conversion to torch dtype: " + str(data_format))


def is_equivalent_data_format(pt_df: torch.dtype, tt_df: DataFormat) -> bool:
    """
    Return True if both PyTorch and Forge data formats are exactly the same
    """

    if tt_df == DataFormat.Float32:
        return pt_df == torch.float32

    if tt_df == DataFormat.Float16:
        return pt_df == torch.float16

    if tt_df == DataFormat.Float16_b:
        return pt_df == torch.bfloat16

    if tt_df == DataFormat.Int8:
        return pt_df == torch.int8

    return False


# def pytorch_tensor_to_tensor_desc(t: torch.Tensor, df: DataFormat = None, element_size=None) -> "PytorchTensorDesc":
#     if isinstance(t, PytorchTensorDesc) or isinstance(t, TilizedTensorDesc):
#         return t

#     if not t.is_contiguous():
#         t = t.contiguous()

#     if df is None:
#         if t.dtype == torch.float32:
#             format = DataFormat.Float32
#         elif t.dtype == torch.bfloat16:
#             format = DataFormat.Float16_b
#         elif t.dtype == torch.float16:
#             format = DataFormat.Float16
#         elif t.dtype == torch.int32:
#             format = DataFormat.Int32
#         elif t.dtype == torch.int8:
#             format = DataFormat.Int8
#         elif t.dtype == torch.int64:
#             logger.warning("Converting int64 to int32 for tilization")
#             t = t.to(torch.int32)   # TODO: Fix this hack
#             format = DataFormat.RawUInt32
#         else:
#             raise RuntimeError("Unsupported torch tensor type for tilization: " + str(t.dtype))
#     else:
#         # If we already know dataformat, don't infer
#         format = df

#         # Before we push the tensors to the queue, we need to make sure that the
#         # tensors are in the right format and aligned between Forge and PyTorch.
#         # If this isn't the case, expected shapes on the queues will be invalid
#         # and the runtime will crash.
#         #
#         # Therefore, when we know the data format, we should check if the tensor
#         # is appropriate/supported PyTorch format. If that isn't the case, we should
#         # convert it to the appropriate PyTorch aligned format.
#         pytorch_dtype = forge_dataformat_to_pytorch_dtype(format)
#         if t.dtype != pytorch_dtype:
#             logger.warning(f"Converting tensor from {t.dtype} to {pytorch_dtype}")
#             t = t.type(pytorch_dtype)

#     tilize_ndim = 4
#     shape = list(t.shape)
#     dim = len(shape)
#     if (dim == 2):
#         dim = 3
#     while len(shape) > tilize_ndim:
#         if shape[0] != 1:
#             raise RuntimeError("Dropping a dimension that's not 1 to reduce shape to 4D: " + str(t.shape))
#         shape = shape[1:]

#     while len(shape) < tilize_ndim:
#         shape = [1] + shape

#     strides = list(t.stride())
#     while len(strides) > tilize_ndim:
#         strides = strides[1:]

#     while len(strides) < tilize_ndim:
#         strides = [strides[0]] + strides

#     if element_size is None:
#         element_size = t.element_size()

#     strides = [s * element_size for s in strides]
#     desc = PytorchTensorDesc(
#         t,
#         element_size,
#         format,
#         dim,
#         shape,
#         strides,
#     )

#     return desc


# def tensor_desc_to_pytorch_tensor(desc: "PytorchTensorDesc") -> torch.Tensor:
#     if desc.format == DataFormat.Float32:
#         dtype = torch.float32
#     elif desc.format == DataFormat.Float16_b:
#         dtype = torch.bfloat16
#     elif desc.format == DataFormat.Float16:
#         dtype = torch.float16
#     elif desc.format == DataFormat.RawUInt32:
#         dtype = torch.int
#     else:
#         raise RuntimeError(f"Unsupported tensor type({desc.format}) for untilization")

#     t = torch.frombuffer(desc, dtype=dtype)
#     t = torch.reshape(t, desc.shape)

#     return t

# def buffer_to_pytorch_tensor(buf_ptr:int, shape: Tuple, format: DataFormat) -> "PytorchTensorDesc":
#     """
#     Convert buffer point to pytorch tensor, given shape and data format.
#     The assumption is that the buffer is in row-major format.
#     """

#     tilize_ndim = 4
#     dim = len(shape)
#     while len(shape) < tilize_ndim:
#         shape = [1] + shape
#     while len(shape) > tilize_ndim:
#         if shape[0] != 1:
#             raise RuntimeError("Trimming a dimension that's not 1")
#         shape = shape[1:]

#     if format == DataFormat.Float32:
#         element_size = 4
#     elif format == DataFormat.Float16_b:
#         element_size = 2
#     elif format == DataFormat.Float16:
#         element_size = 2
#     else:
#         raise RuntimeError("Unsupported format")

#     strides = [element_size]
#     for i in range(tilize_ndim-1):
#         strides = [shape[-1-i] * strides[0]] + strides

#     desc = PytorchTensorDesc(
#         buf_ptr,
#         element_size,
#         format,
#         dim,
#         shape,
#         strides,
#     )

#     return tensor_desc_to_pytorch_tensor(desc)


def pad_sparse_pytorch_tensor_to_forge(sparse: torch.Tensor) -> torch.Tensor:
    assert len(sparse.shape) == 4, "This function should only be invoked from pad_pytorch_tensor_to_forge"
    sparse_r = align_up_tile(sparse.shape[-2])
    sparse_c = align_up_tile(sparse.shape[-1])
    sparse = sparse.coalesce()
    return torch.sparse_coo_tensor(
        sparse.indices(),
        sparse.values(),
        (sparse.shape[0], sparse.shape[1], sparse_r, sparse_c),
        dtype=sparse.dtype,
    )


def is_forge_shape(tensor: torch.Tensor, min_dim=-1) -> bool:
    dim_ok = len(tensor.shape) >= min_dim if min_dim > 0 else len(tensor.shape) == 4
    if not dim_ok or len(tensor.shape) > 4:
        return False
    return tensor.shape[-1] % TILE_DIM == 0 and tensor.shape[-2] % TILE_DIM == 0


def pad_pytorch_tensor_to_forge(
    tensor: torch.Tensor,
    tile_broadcast_dims: List[int],
    squeeze: bool = False,
    microbatch=1,
    tile_r=TILE_DIM,
    tile_c=TILE_DIM,
) -> torch.Tensor:
    """
    Pad pytorch tensor to 4D tile-snapped dimensions. Broadcast given dims to tile size.
    """
    # Skip padding if not needed
    min_dim = 2 if squeeze and tensor.shape[0] != microbatch and len(tensor.shape) > 2 else 4
    if is_forge_shape(tensor, min_dim):
        return tensor

    new_tensor = tensor

    while len(new_tensor.shape) < min_dim:
        if new_tensor.shape[0] == microbatch:
            new_tensor = new_tensor.unsqueeze(1)
        else:
            new_tensor = new_tensor.unsqueeze(0)

    while len(new_tensor.shape) > 5:
        assert new_tensor.shape[0] == 1, "Invalid dimension size above dim 5"
        new_tensor = new_tensor.squeeze(0)

    if new_tensor.shape[0] < microbatch:
        new_tensor = new_tensor.broadcast_to([microbatch, *new_tensor.shape[1:]]).contiguous()

    if is_forge_shape(new_tensor):
        new_tensor = new_tensor.detach().contiguous()
        return new_tensor  # done after adjusting number of dims

    if new_tensor.is_sparse:
        return pad_sparse_pytorch_tensor_to_forge(new_tensor)

    def get_pad(dim, tile_dim):
        if dim % tile_dim > 0:
            return tile_dim - (dim % tile_dim)
        return 0

    dim_c = new_tensor.shape[-1]
    dim_r = new_tensor.shape[-2]

    pad_right = get_pad(dim_c, tile_c)
    pad_bottom = get_pad(dim_r, tile_r)

    # broadcast to tile dim if told to do so
    mode_right = "replicate" if (-1 in tile_broadcast_dims or 3 in tile_broadcast_dims) else "constant"
    mode_bottom = "replicate" if (-2 in tile_broadcast_dims or 2 in tile_broadcast_dims) else "constant"

    if mode_right == "replicate":
        assert dim_c == 1, "Trying to broadcast to tile dim a dimension that's not 1"

    if mode_bottom == "replicate":
        assert dim_r == 1, "Trying to broadcast to tile dim a dimension that's not 1"

    # Torch pad needs at least 3d, and float32
    original_type = new_tensor.dtype
    if original_type != torch.float32:
        new_tensor = new_tensor.type(torch.float32)
    original_dim = len(new_tensor.shape)
    while len(new_tensor.shape) < 3:
        new_tensor = new_tensor.unsqueeze(0)
    ret = torch.nn.functional.pad(new_tensor, (0, pad_right, 0, 0), mode=mode_right)
    ret = torch.nn.functional.pad(ret, (0, 0, 0, pad_bottom), mode=mode_bottom)
    while len(ret.shape) > original_dim:
        ret = ret.squeeze(0)

    if ret.dtype != original_type:
        ret = ret.type(original_type)

    ret = ret.detach()
    ret.requires_grad = tensor.requires_grad
    return ret


def narrow_forge_tensor_to_pytorch(
    tensor: torch.Tensor, shape: List[int], has_microbatch_dim: bool = False
) -> torch.Tensor:
    """
    Narrow 4D / tile-snapped tensor to original pytorch shape
    """
    if tensor.is_sparse:
        tensor = tensor.coalesce()
        values = tensor.values()
        indices = tensor.indices().tolist()
        if len(tensor.shape) > 2 and len(shape) == 2:
            if all([all([i == 0] for i in idxes) for idxes in indices[:-2]]):
                indices = indices[-2:]

        return torch.sparse_coo_tensor(
            indices,
            values,
            (shape),
            dtype=tensor.dtype,
        )

    new_tensor = tensor

    def volume(shape):
        return reduce(mul, shape)

    # case with 0D shape implies this is a scalar that we padded to tile shape
    if len(shape) == 0:
        new_tensor = new_tensor.narrow(-1, 0, 1).narrow(-2, 0, 1).squeeze()
        return new_tensor

    # case with 1D shape implies we have a vector that we padded to tile shape
    elif len(shape) == 1:
        if len(new_tensor.shape) == 1:
            new_tensor = new_tensor.narrow(0, 0, shape[0])
        elif len(new_tensor.shape) > 2 and new_tensor.shape[0] == shape[0]:
            new_tensor = new_tensor.narrow(-1, 0, 1).narrow(-2, 0, 1)
        else:
            new_tensor = new_tensor.narrow(-1, 0, shape[0]).narrow(-2, 0, 1)

    # case with 2+D shape implies that we have matrices that we padded to tile shapes
    elif volume(shape) != volume(new_tensor.shape):
        # case when tensor shape differs due to the attached golden transform during
        # some of the passes (e.g. optimization graph pass). Working on T dim operands
        # as rest of them are potentially padded which differs anyhow
        transpose_before_tile_narrow = False
        if (len(new_tensor.shape) == 4 and len(shape) == 4) and (new_tensor.shape[-3] != shape[-3]):
            transpose_before_tile_narrow = True

        # Transpose before narrow
        if transpose_before_tile_narrow:
            new_tensor = new_tensor.transpose(-3, -2)

        # If we are narrowing an activation, we assume index 0 is the batch dimension,
        # so we of course do not want to narrow that. If we are narrowing a parameter,
        # then index 0 is of course not a batch dimension, so we don't care if
        # we have to narrow that
        if has_microbatch_dim:
            shape_ = shape[1:]
        else:
            shape_ = shape

        # Remove the padding along the 'height' dimension
        if len(shape_) == 1:
            new_tensor = new_tensor.narrow(-2, 0, 1)
        else:
            new_tensor = new_tensor.narrow(-2, 0, shape_[-2])

        # Remove the padding along the 'width' dimension
        new_tensor = new_tensor.narrow(-1, 0, shape_[-1])

        # Transpose to the original shape
        if transpose_before_tile_narrow:
            new_tensor = new_tensor.transpose(-3, -2)

    # reshape the rest
    new_tensor = new_tensor.reshape(shape)

    return new_tensor


def change_rank(tensor: torch.Tensor, rank: int):
    while len(tensor.shape) > rank:
        assert tensor.shape[0] == 1
        tensor = tensor.select(0, 0)
    while len(tensor.shape) < rank:
        tensor = tensor.unsqueeze(0)
    return tensor


def to_tf_variables(
    tensors: Tuple[Union[torch.Tensor, Tensor], ...], convert_format: bool = False
) -> Tuple[tf.Variable, ...]:
    """
    Take a tuple of either pytorch, TF or forge tensors, and return TF Variables.
    """
    tf_variables = []
    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)
    for t in tensors:
        if isinstance(t, torch.Tensor):
            tf_variables.append(
                tf.Variable(
                    tf.convert_to_tensor(
                        value=t.detach().numpy(),
                        dtype=map_pt_dtype_to_tf(t.dtype),
                    ),
                    trainable=t.requires_grad,
                )
            )
        elif isinstance(t, Tensor):
            pt_value = t.value() if t.has_value() else create_pt_zeros(t)
            # TODO: Generalize
            if pt_value.dtype == torch.bfloat16:
                pt_value = copy.deepcopy(pt_value.detach()).float()
            tf_variables.append(
                tf.Variable(
                    tf.convert_to_tensor(
                        value=pt_value.detach().numpy(),
                        dtype=map_pt_dtype_to_tf(pt_value.dtype),
                    ),
                    trainable=t.value().requires_grad,
                )
            )
        elif isinstance(t, tf.Tensor):
            tf_variables.append(tf.Variable(t))
        elif isinstance(t, tf.Variable):
            tf_variables.append(t)
        elif t is None:
            tf_variables.append(None)
        else:
            raise RuntimeError("Unknown type of tensor")

    return tuple(tf_variables)


def to_tf_tensors(
    tensors: Union[
        Tuple[Union[torch.Tensor, Tensor, tf.Tensor], ...], Dict[str, Union[torch.Tensor, Tensor, tf.Tensor]]
    ],
    convert_format: bool = False,
    force_float32: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Take a tuple of either tensorflow or forge tensors, and return pytorch tensors. Generate zero-tensors
    if no value exists.
    """
    tf_tensors = []
    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)
    for t in tensors:
        if isinstance(t, (tf.Tensor, tf.Variable)):
            assert (
                not convert_format
            ), "Can't convert format of raw pytorch tensor - don't know what the target format is"
            tf_tensors.append(t)
        elif isinstance(t, torch.Tensor):
            # TODO: do we always set 'requires grad'? requires_grad will be set true if the type is differentiable
            requires_grad = torch.is_complex(t) or torch.is_floating_point(t)
            if force_float32:
                t = t.float() if t.dtype.is_floating_point else t

            tensor = tf.convert_to_tensor(
                t.detach().float().numpy() if t.dtype == torch.bfloat16 else t.detach().numpy(),
                map_pt_dtype_to_tf(t.dtype),
            )
            tensor.requires_grad = requires_grad
            tf_tensors.append(tensor)
        elif isinstance(t, Tensor):
            if convert_format:
                t = t.to_format(t.data_format)
            if t.has_value():
                tf_tensors.append(t.value())
            else:
                tf_tensors.append(create_pt_zeros(t))
        elif t is None:
            tf_tensors.append(None)
        elif isinstance(t, (list, tuple)):
            tf_tensors.append(to_tf_tensors(t))
        elif isinstance(t, dict):
            tf_tensor_list = to_tf_tensors(list(t.values()))
            tf_dict = {k: v for (k, _), v, in zip(t.items(), tf_tensor_list)}
            tf_tensors.append(tf_dict)
        else:
            raise RuntimeError("Unknown type of tensor")

    ret = tuple(tf_tensors) if isinstance(tensors, (tuple, list)) else (tf_tensors,)
    return ret


def create_pt_zeros(tensor: Tensor) -> torch.Tensor:
    # generate zeros
    pt = torch.zeros(size=tensor.shape.get_pytorch_shape(), dtype=forge_dataformat_to_pytorch_dtype(tensor.data_format))
    pt.requires_grad = tensor.requires_grad if isinstance(tensor, TensorFromPytorch) else False
    return pt


def to_pt_tensors(tensors: Union[AnyTensor, Tuple[AnyTensor, ...], List[AnyTensor]]) -> Tuple[torch.Tensor, ...]:
    pytorch_tensors: List[torch.Tensor] = []

    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)

    for t in tensors:
        pytorch_tensors.append(to_pt_tensor(t))

    return tuple(pytorch_tensors)


def to_pt_tensor(t: AnyTensor) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t
    elif isinstance(t, (tf.Tensor, tf.Variable)):
        pt = torch.Tensor(t.numpy() if t.dtype != tf.bfloat16 else tf.cast(t, tf.float32).numpy()).type(
            map_tf_dtype_to_pt(t.dtype)
        )
        pt.requires_grad = (
            t.trainable if isinstance(t, tf.Variable) else torch.is_complex(pt) or torch.is_floating_point(pt)
        )
        return pt
    elif isinstance(t, Tensor):
        assert t.has_value(), "Expected Forge tensor to have a value"
        return t.value()
    elif isinstance(t, paddle.Tensor):
        pt = torch.Tensor(t.numpy()).type(map_pd_dtype_to_pt(t.dtype))
        pt.requires_grad = t.stop_gradient == False
        return pt
    elif isinstance(t, np.ndarray):
        return torch.from_numpy(t)
    elif isinstance(t, jax.Array):
        return torch.from_numpy(np.array(t))
    else:
        raise RuntimeError(f"Unknown type of tensor: {type(t)}")


def to_pd_tensors(tensors: Union[AnyTensor, Tuple[AnyTensor, ...], List[AnyTensor]]) -> Tuple[paddle.Tensor, ...]:
    paddle_tensors = []

    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)

    for t in tensors:
        paddle_tensors.append(to_pd_tensor(t))

    return tuple(paddle_tensors)


def to_pd_tensor(pt: torch.Tensor) -> paddle.Tensor:
    if isinstance(pt, paddle.Tensor):
        return pt
    elif isinstance(pt, torch.Tensor):
        pd = paddle.to_tensor(pt.detach().numpy(), dtype=map_pt_dtype_to_pd(pt.dtype))
        pd.stop_gradient = not pt.requires_grad
        return pd
    else:
        raise RuntimeError(f"Unsupported type of tensor: {type(pt)}")


def to_jax_tensors(
    tensors: Union[
        Tuple[Union[torch.Tensor, Tensor, tf.Tensor], ...], Dict[str, Union[torch.Tensor, Tensor, tf.Tensor]]
    ],
    convert_format: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Take a tuple of either pytorch or forge tensors, and return pytorch tensors. Generate zero-tensors
    if no value exists.
    """
    jax_tensors = []

    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)
    for t in tensors:
        if isinstance(t, torch.Tensor):
            assert (
                not convert_format
            ), "Can't convert format of raw pytorch tensor - don't know what the target format is"
            jax_tensors.append(jnp.asarray(t.detach().numpy()))
        elif isinstance(t, (tf.Tensor, tf.Variable)):
            jax_tensor = jnp.asarray(t.numpy())
            jax_tensors.append(jax_tensor)
        elif isinstance(t, Tensor):
            if convert_format:
                t = t.to_format(t.data_format)
            if t.has_value():
                jax_tensors.append(t.value().detach().numpy())
            else:
                jax_tensors.append(create_pt_zeros(t).detach().numpy())
        elif t is None:
            jax_tensors.append(None)
        elif isinstance(t, (list, tuple)):
            jax_tensors.append(to_jax_tensors(t))
        elif isinstance(t, dict):
            jax_tensor_list = to_jax_tensors(list(t.values()))
            jax_dict = {k: v for (k, _), v, in zip(t.items(), jax_tensor_list)}
            jax_tensors.append(jax_dict)

        elif isinstance(t, np.ndarray):
            jax_tensors.append(jnp.asarray(t))
        elif isinstance(t, jax.Array):
            jax_tensors.append(t)
        else:
            raise RuntimeError(f"Unknown type of tensor: {type(t)}")

    ret = tuple(jax_tensors) if isinstance(tensors, (tuple, list)) else (jax_tensors,)
    return ret


def to_forge_tensors(tensors: Tuple[Union[torch.Tensor, Tensor], ...]) -> Tuple[Tensor, ...]:
    """
    Take a tuple of either pytorch or forge tensors, and return forge tensors
    """
    forge_tensors = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            forge_tensors.append(Tensor.create_from_torch(t))
        elif isinstance(t, Tensor):
            forge_tensors.append(t)
        elif isinstance(t, (tf.Tensor, tf.Variable)):
            pt = torch.Tensor(t.numpy()).type(map_tf_dtype_to_pt(t.dtype))
            pt.requires_grad = (
                t.trainable if isinstance(t, tf.Variable) else torch.is_complex(pt) or torch.is_floating_point(pt)
            )
            forge_tensors.append(Tensor.create_from_torch(pt))
        elif t is None:
            continue
        elif isinstance(t, (list, tuple)):
            forge_tensors.append(to_forge_tensors(t))
        elif isinstance(t, dict):
            tensor_list = list(t.values())
            pb_dict = {k: v for (k, _), v, in zip(t.items(), to_forge_tensors(tensor_list))}
            forge_tensors.append(pb_dict)
        else:
            raise RuntimeError(f"Unknown type of tensor: {type(t)}")

    return tuple(forge_tensors)


def remove_microbatch(tensors: Tuple[Union[torch.Tensor, Tensor], ...]) -> Tuple[torch.Tensor, ...]:
    out = []
    for input in tensors:
        if isinstance(input, torch.Tensor):
            out.append(Tensor.create_from_torch(torch.narrow(input.clone(), 0, 0, 1)))
        elif isinstance(input, (tf.Variable, tf.Tensor)):
            torch_tensor = torch.Tensor(input.numpy()).type(map_tf_dtype_to_pt(input.dtype))
            out.append(Tensor.create_from_torch(torch.narrow(torch_tensor, 0, 0, 1)))
        elif isinstance(input, (list, tuple)):
            out.append(remove_microbatch(input))
        elif isinstance(input, dict):
            tensor_list = list(input.values())
            out_dict = {k: v for (k, _), v, in zip(input.items(), remove_microbatch(tensor_list))}
            out.append(out_dict)
        else:
            out.append(
                Tensor.create_from_torch(
                    torch.narrow(input.value().clone(), 0, 0, 1), dev_data_format=input.data_format
                )
            )

    return tuple(out)


def get_constant_inputs(
    constant_nodes, device_constant_and_parameters, consteval_trace, name: str, is_forge: bool
) -> Dict[str, torch.Tensor]:

    consteval_graph = consteval_trace.get(name, None)

    if consteval_graph is None:
        if name in constant_nodes:
            return {name: get_constant_input_value(constant_nodes[name], is_forge)}
        return {name: device_constant_and_parameters[name]}

    values = {}
    for node_name in consteval_graph["topological_sorted_nodes"]:
        node = consteval_graph["nodes"][node_name]
        if node["opcode"] == "Input":
            if node_name in constant_nodes:
                values[node_name] = get_constant_input_value(constant_nodes[node_name], is_forge)
            else:
                values[node_name] = device_constant_and_parameters[node_name]
    return values


def consteval_tensor(consteval_trace, name: str, inputs: Dict[str, torch.Tensor], is_forge: bool) -> torch.Tensor:
    import forge.op.eval.forge as eval_module

    consteval_graph = consteval_trace.get(name, None)

    if consteval_graph is None:
        return inputs[name]

    def eval_op(op_type, inputs):
        forge_eval = eval_module.get_f_forge_eval(OpType(op_type["type"], op_type["attrs"], op_type["named_attrs"]))
        return forge_eval(inputs)

    logger.debug("ConstEval graph: {}", name)
    node_to_tensor: Dict[str, torch.Tensor] = {}
    output: Optional[torch.Tensor] = None
    tile_r, tile_c = (TILE_DIM, TILE_DIM)

    for node_name in consteval_graph["topological_sorted_nodes"]:
        node = consteval_graph["nodes"][node_name]
        if node["opcode"] == "Input":
            input_value = inputs[node_name]

            if is_forge:
                input_value = narrow_forge_tensor_to_pytorch(
                    input_value, node["cache"]["shape"], has_microbatch_dim=False
                )

            node_to_tensor[node_name] = input_value
        elif node["opcode"] in {"ForgeOp", "ForgeOp"}:
            inputs_after_tms: List[torch.Tensor] = []
            for input_index, operand in enumerate(node["input_nodes"]):
                operand_tensor = node_to_tensor[operand]
                if node.get("input_tms", None):
                    for tm in node["input_tms"][input_index]:
                        operand_tensor = eval_op(tm["op_type"], [operand_tensor])
                inputs_after_tms.append(operand_tensor)

            output = eval_op(node["op_type"], inputs_after_tms)
            node_to_tensor[node_name] = output

        elif node["opcode"] == "Output":
            output = node_to_tensor[node["input_nodes"][0]]

    assert output is not None, "Expect a valid tensor output out of consteval"
    if is_forge:
        output = pad_pytorch_tensor_to_forge(output, [], tile_r=tile_r, tile_c=tile_c)
    return output


def consteval_input(consteval_trace, name: str, inputs: Dict[str, torch.Tensor], is_forge: bool) -> torch.Tensor:
    const_eval_tensor = consteval_tensor(consteval_trace, name, inputs, is_forge)
    # This: "torch.empty(const_eval_tensor.shape).copy_(const_eval_tensor)" will create tensor with contiguous memory layout consistent with its current shape.
    # We are doing this because constant input tensors should have memory layout consistent with their shape.
    # Sometimes, the stride is inconsistent with shape because some consteval operations might change the shape but not the stride.
    # For example, if we had transpose in consteval graph, output tensor would have stride same as input.
    # However, since we store that input as constant tensor, its shape defines its stride.
    return torch.empty(const_eval_tensor.shape, dtype=const_eval_tensor.dtype).copy_(const_eval_tensor)


def consteval_shape(compiled_graph_state, name: str, tensor: torch.Tensor, is_forge: bool = False) -> torch.Tensor:
    consteval_graph = compiled_graph_state.consteval_trace.get(name, None)
    if consteval_graph is None:
        return tensor.shape

    for node_name in consteval_graph["topological_sorted_nodes"]:
        node = consteval_graph["nodes"][node_name]
        if node["opcode"] == "Output":
            return node["cache"]["shape"]
    assert False, "No output node found in consteval graph"


def compare_tensors(t0, t1):
    return torch.equal(t0, t1)


def const_eval_tensor(inputs, consteval_trace, input_name, is_forge=True):
    contains_recorded_operations = consteval_trace[input_name]
    if contains_recorded_operations:
        value = detach_tensors(
            [consteval_input(consteval_trace, input_name, inputs, is_forge)], fix_non_contiguos=True
        )[0]
    else:
        value = pad_pytorch_tensor_to_forge(inputs[input_name], []) if is_forge else inputs[input_name]
    # cast if necessary
    forge_dtype = pytorch_dtype_to_forge_dataformat(value.dtype)
    if value.dtype != forge_dataformat_to_pytorch_dtype(forge_dtype):
        value = value.to(forge_dataformat_to_pytorch_dtype(forge_dtype))
    return value


def get_device_constant_and_parameters(
    device, *, constant_to_tensor=None, updated_parameter_values=None
) -> Dict[str, torch.Tensor]:
    inputs = {}
    if updated_parameter_values is None:
        for p in device.get_parameters(ignore_unused_parameters=False):
            value = p.value(is_forge=False)
            if value is None:
                raise RuntimeError(f"Parameter {p.get_name()} has not value set.")
            inputs[p.get_name()] = value
    else:
        for parameter, value in updated_parameter_values.items():
            inputs[parameter] = value

    if constant_to_tensor is not None:
        for name, value in constant_to_tensor.items():
            inputs[name] = value
    return inputs


def get_post_const_eval_tensors(
    graph, device_constant_and_parameters, consteval_trace, ordered_input_names, is_forge=True
) -> Dict[str, torch.Tensor]:
    post_const_eval_constants: Dict[str, torch.Tensor] = {}

    constant_nodes = {node.name: node for node in graph.get_constant_nodes(recurse=True)}

    for input_name in ordered_input_names:
        # Load input constant tensors for consteval
        inputs = get_constant_inputs(
            constant_nodes, device_constant_and_parameters, consteval_trace, input_name, is_forge
        )

        post_const_eval_constants[input_name] = const_eval_tensor(
            inputs,
            consteval_trace,
            input_name,
            is_forge,
        )

    return post_const_eval_constants


# def _embedding_index(tensor: torch.Tensor, original_shape: Tuple[int, ...], queue: DramIODesc):
#     assert queue.data_format in [DataFormat.RawUInt8, DataFormat.RawUInt16, DataFormat.RawUInt32]
#     assert len(tensor.shape) <= 2, "Must be a 1d tensor"
#     assert len(original_shape) <= 1 or original_shape[-2] == 1, "Must be a 1d tensor"
#     assert len(original_shape) <= 2 or original_shape[-3] == 1, "Must be a 1d tensor"

#     q_rt = queue.bufq_grid_dim_r * queue.mblock_m * queue.ublock_rt
#     w = tensor.shape[0] if len(tensor.shape) > 1 else 1
#     pad = align_up(tensor.shape[-1], TILE_DIM) - tensor.shape[-1]
#     tensor = torch.nn.functional.pad(tensor, (0, pad))
#     tensor = tensor.reshape(w, 1, 1, tensor.shape[-1])
#     tensor[:, :, :, original_shape[-1]:] = ~torch.tensor(0, dtype=tensor.dtype)
#     tensor = tensor.view(w, q_rt, -1, TILE_DIM)
#     pad = align_up(tensor.shape[-2], TILE_DIM) - tensor.shape[-2]
#     tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad))
#     tensor = tensor.view(w, q_rt, -1, TILE_DIM, TILE_DIM)
#     tensor = tensor.transpose(2, 3).view(w, 1, q_rt * TILE_DIM, -1)

#     assert len(tensor.shape) == 4, "_embedding_index: rank changed"
#     assert tensor.shape[0] == w, "_embedding_index: w changed"
#     assert tensor.shape[1] == queue.t, "_embedding_index: t changed"
#     assert tensor.shape[2] == (queue.bufq_grid_dim_r * queue.mblock_m * queue.ublock_rt * TILE_DIM), "_embedding_index: tensor dims mismatch q dims"
#     assert tensor.shape[3] == (queue.bufq_grid_dim_c * queue.mblock_n * queue.ublock_ct * TILE_DIM), "_embedding_index: tensor dims mismatch q dims"
#     return tensor

# def _reinterpret_shape(tensor: torch.Tensor, shape: List[int], queue: DramIODesc, tile_bcast_dims: List[int]):
#     tensor = tensor.contiguous().view(shape)
#     tile_r = queue.tile_height
#     tile_c = queue.tile_width
#     microbatch = queue.input_count
#     tensor = pad_pytorch_tensor_to_forge(tensor, tile_bcast_dims, squeeze=True, microbatch=microbatch, tile_r=tile_r, tile_c=tile_c)
#     return tensor, queue

# def _prestride_shape(tensor: torch.Tensor, stride_height: int, stride_width: int, queue: DramIODesc):
#     assert stride_height == stride_width, "Backend supports only square strides for prestriding transform"
#     stride = stride_height
#     stride_desc = StrideDescriptor()
#     stride_desc.stride = stride
#     stride_desc.xy_offsets = [(x, y) for y in range(stride) for x in range(stride)]
#     queue.s_descriptor = stride_desc
#     return tensor, queue


def do_runtime_transform(transform, tensor, q, tile_bcast_dims):
    assert False
    if transform.type == RuntimeTensorTransformType.EmbeddingIndex:
        return _embedding_index(tensor, transform.original_shape, q), q
    elif transform.type == RuntimeTensorTransformType.ReinterpretShape:
        return _reinterpret_shape(tensor, transform.reinterpreted_shape.as_list(), q, tile_bcast_dims)
    elif transform.type == RuntimeTensorTransformType.NoTransform:
        return tensor, q
    elif transform.type == RuntimeTensorTransformType.Prestride:
        return _prestride_shape(tensor, transform.stride_height, transform.stride_width, q)
    else:
        assert False, f"Unsupported runtime transform type: {transform.type}"


def eval_runtime_transform(transform, inp, q, tile_bcast_dims):
    if isinstance(transform, str):
        transform = json.loads(transform)
        transform = RuntimeTensorTransform.from_json(transform)
    logger.info(f"Aplying runtime transform {transform}")
    return do_runtime_transform(transform, inp, q, tile_bcast_dims)
