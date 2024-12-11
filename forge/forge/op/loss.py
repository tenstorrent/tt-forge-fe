# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from forge.op.tm import Broadcast, Unsqueeze
from ..module import ForgeModule
from .constant import Constant
from .eltwise_unary import Log, Abs
from .eltwise_binary import Add, GreaterEqual, Less, Subtract, Multiply
from .nn import Softmax
from .reduce import ReduceSum, ReduceAvg


class CrossEntropyLoss(ForgeModule):
    """
    Cross-Entropy Loss (with mean reduction)

    loss = reduce_avg(-1 * sum(labels * log(softmax(predictions)), dim=-1), dim=0)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.is_loss = True

    def forward(self, predictions, labels):
        assert predictions.ndim() == 2, f"Predictions must be a 2D tensor. Got {predictions.ndim()}."
        assert (
            predictions.shape == labels.shape
        ), f"Shapes of predictions and labels must match. predictions.shape={predictions.shape} labels.shape={labels.shape}."

        softmax = Softmax("softmax", predictions, dim=-1)
        log_softmax = Log("log", softmax)

        product = Multiply("products", labels, log_softmax)
        log_loss = ReduceSum("log_loss", product, dim=-1)

        negative_one_constant = Constant("negative_one_const", constant=-1.0)
        negative_log_loss = Multiply(
            "negative_log_loss",
            log_loss,
            negative_one_constant,
        )

        reduction_avg = ReduceAvg("reduction_avg", negative_log_loss, dim=0)
        return reduction_avg


class L1Loss(ForgeModule):
    """
    L1Loss

    L1Loss is abs(labels - prediction), optionally reduced using ReduceAvg(default) or ReduceSum.
    """

    def __init__(self, name: str, reduction: str = "avg"):
        super().__init__(name)
        self.reduction = reduction
        self.is_loss = True

    def forward(self, prediction, labels):
        diff = Abs("abs", Subtract("sub", prediction, labels))

        if self.reduction == "none":
            return diff

        # TODO: z reduce?
        if self.reduction == "avg":
            r_reduce = ReduceAvg("r_avg", diff, -2)
            c_reduce = ReduceAvg("c_avg", r_reduce, -1)
            return c_reduce

        if self.reduction == "sum":
            r_reduce = ReduceSum("r_avg", diff, -2)
            c_reduce = ReduceSum("c_avg", r_reduce, -1)
            return c_reduce

        raise RuntimeError("Unsupported reduce type: " + self.reduction)


class HuberLoss(ForgeModule):
    def __init__(self, name: str, delta: float = 1.0, reduction: str = "avg"):
        super().__init__(name)
        self.delta = delta
        self.reduction = reduction
        self.is_loss = True

    def forward(self, prediction, labels):
        diff = Subtract("sub", prediction, labels)
        abs_diff = Abs("abs", diff)

        # delta
        const_delta = Constant("delta", constant=self.delta)
        const_delta = Unsqueeze("cast_delta", const_delta, dim=0)
        broadcast_delta = Broadcast("broadcast_delta_1", const_delta, dim=0, shape=abs_diff.shape[0])
        delta = Broadcast("broadcast_delta_2", broadcast_delta, dim=1, shape=abs_diff.shape[1])

        lt_delta = Less("lt_delta", abs_diff, delta)
        ge_delta = GreaterEqual("ge_delta", abs_diff, delta)

        # 0.5 * (x - y)**2
        square = Multiply("square", diff, diff)
        half_const = Constant("half", constant=0.5)
        half_const = Unsqueeze("half_unsqueeze", half_const, dim=0)
        broadcast_half = Broadcast("broad_cast_half_dim0", half_const, dim=0, shape=square.shape[0])
        half = Broadcast("broad_cast_half_dim1", broadcast_half, dim=1, shape=square.shape[1])
        half_square = Multiply("half_square", half, square)

        # delta * (abs_diff - 0.5 * delta)
        half_delta = Multiply("half_delta", half, delta)
        delta_diff = Subtract("delta_diff", abs_diff, half_delta)
        mul_delta_diff = Multiply("mul_delta_diff", delta, delta_diff)

        loss_lt = Multiply("loss_lt", lt_delta, half_square)
        loss_ge = Multiply("loss_ge", ge_delta, mul_delta_diff)

        loss = Add("loss", loss_lt, loss_ge)

        if self.reduction == "avg":
            r_reduced = ReduceAvg("r_avg", loss, 0)
            c_reduced = ReduceAvg("c_avg", r_reduced, 1)
            return c_reduced
        elif self.reduction == "sum":
            r_reduced = ReduceSum("r_sum", loss, 0)
            c_reduced = ReduceSum("c_sum", r_reduced, 1)
            return c_reduced

        return RuntimeError("Unsupported reduce type: " + self.reduction)
