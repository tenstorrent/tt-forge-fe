# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from typing import List, Tuple, Dict, Union, Optional
from forge._C.graph import NodeContext, OpType
from forge._C.passes import LoweringContext, DecomposingContext
from forge._C.autograd import AutogradContext


class OpTypeWrapper:
    """
    Utility wrapper class that abstracts the underlying C++ object OpType
    """

    def __init__(self, op_type: Union[str, OpType]):
        if type(op_type) is str:
            object.__setattr__(self, "op_type", OpType(op_type))
        elif type(op_type) is OpType:
            object.__setattr__(self, "op_type", op_type)
        else:
            cls_name = self.__getattribute__("__class__").__name__
            raise RuntimeError(f"Called {cls_name} __init__ fn, did you mean to call '{cls_name}.create'?")

    def __getattr__(self, name):
        try:
            return getattr(self.op_type, name)
        except IndexError:
            cls_name = self.__getattribute__("__class__").__name__
            raise AttributeError(f"'{cls_name}' object has no attribute '{name}' (via OpType cpp underlying class)")

    def __setattr__(self, name, value):
        return setattr(self.op_type, name, value)

    def __repr__(self):
        return repr(self.op_type)


class PyOp(OpTypeWrapper):
    """
    Forge IR Op interface

    All Forge IR ops must inherit and implement this interface
    """

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Derived classes can declare their create method however
        they please, but they must use OpTypeWrapper base class
        constructor to construct themselves

        For example:

        class Transpose(PyOp):
            @classmethod
            def create(cls, dim0, dim1):
                self = cls("transpose") # <- OpTypeWrapper base class __init__
                self.dim0 = dim0        # <- set attributes
                self.dim1 = dim1
                return self             # <- Return self
        """
        raise NotImplemented()

    def eval(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplemented()

    def shape(self, tensor_shapes: List[Tuple[int]]) -> Tuple[Tuple[int], List[int]]:
        raise NotImplemented()

    def backward(
        self,
        autograd_context: AutogradContext,
        operand_idx: int,
        tensors: List[NodeContext],
        output: NodeContext,
        grad: NodeContext,
    ):
        raise NotImplemented()

    def lower(
        self,
        lowering_context: LoweringContext,
        tensors: List[NodeContext],
        outputs: List[NodeContext],
    ):
        raise NotImplemented()

    def decompose(self, decomposing_context: DecomposingContext, tensors: List[NodeContext]):
        # Optional implementation
        pass

    def decompose_post_autograd(self, decomposing_context: DecomposingContext, tensors: List[NodeContext]):
        # Optional implementation
        pass

    def decompose_post_optimize(self, decomposing_context: DecomposingContext, tensors: List[NodeContext]):
        # Optional implementation
        pass

    def initial_flops_estimate(self, tensor_shapes: List[Tuple[int]]) -> int:
        # Optional implementation
        pass

    def is_tm(self) -> bool:
        raise NotImplemented()

    def is_eltwise(self) -> bool:
        raise NotImplemented()

    def is_eltwise_binary(self) -> bool:
        raise NotImplemented()

    def is_eltwise_unary(self) -> bool:
        raise NotImplemented()

    def is_eltwise_nary(self) -> bool:
        raise NotImplemented()


class PyTM(PyOp):
    """
    Forge IR TM interface

    All Forge IR tms must inherit and implement this interface
    """

    def is_tm(self) -> bool:
        return True

    def is_eltwise(self) -> bool:
        return False

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False


class PyEltwiseOp(PyOp):
    def is_tm(self) -> bool:
        return False

    def is_eltwise(self) -> bool:
        return True

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False


class PyEltwiseBinaryOp(PyEltwiseOp):
    def is_eltwise_binary(self) -> bool:
        return True


class PyEltwiseUnaryOp(PyEltwiseOp):
    def is_eltwise_unary(self) -> bool:
        return True


class PyEltwiseNaryOp(PyEltwiseOp):
    def is_eltwise_nary(self) -> bool:
        return True


class ForgeOp(OpTypeWrapper):
    """
    Forge IR Op interface

    All Forge IR ops must inherit and implement this interface
    """

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Derived classes can declare their create method however
        they please, but they must use OpTypeWrapper base class
        constructor to construct themselves

        For example:

        class Transpose(ForgeOp):
            @classmethod
            def create(cls, dim0, dim1):
                self = cls("transpose") # <- OpTypeWrapper base class __init__
                self.dim0 = dim0        # <- set attributes
                self.dim1 = dim1
                return self             # <- Return self
        """
        raise NotImplemented()

    def eval(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplemented()

    def shape(self, tensor_shapes: List[Tuple[int]], tile_height: int, tile_width: int) -> Tuple[Tuple[int], List[int]]:
        raise NotImplemented()

    def parallelization(self, op_shape, fracture_factor) -> Tuple[int]:
        raise NotImplemented()

    def input_ublock_order(self, num_tensors: int):
        raise NotImplemented()

    def execution_cycles(self, arch_name, op_model) -> int:
        raise NotImplemented()

    def is_tm(self) -> bool:
        raise NotImplemented()

    def is_eltwise(self) -> bool:
        raise NotImplemented()

    def is_eltwise_binary(self) -> bool:
        raise NotImplemented()

    def is_eltwise_unary(self) -> bool:
        raise NotImplemented()

    def is_eltwise_nary(self) -> bool:
        raise NotImplemented()


class ForgeTM(ForgeOp):
    """
    Forge IR TM interface

    All Forge IR tms must inherit and implement this interface
    """

    def is_tm(self) -> bool:
        return True

    def is_eltwise(self) -> bool:
        return False

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False


class ForgeEltwiseOp(ForgeOp):
    def is_tm(self) -> bool:
        return False

    def is_eltwise(self) -> bool:
        return True

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False


class ForgeEltwiseBinaryOp(ForgeEltwiseOp):
    def is_eltwise_binary(self) -> bool:
        return True


class ForgeEltwiseUnaryOp(ForgeEltwiseOp):
    def is_eltwise_unary(self) -> bool:
        return True


class ForgeEltwiseNaryOp(ForgeEltwiseOp):
    def is_eltwise_nary(self) -> bool:
        return True
