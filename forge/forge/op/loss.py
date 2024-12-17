# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..module import ForgeModule
from .constant import Constant
from .eltwise_unary import Log, Abs
from .eltwise_binary import Subtract, Multiply
from .nn import Softmax
from .reduce import ReduceSum, ReduceAvg


class CrossEntropyLoss(ForgeModule):
    """
    Cross-Entropy Loss (with mean reduction)

    loss = reduce_avg(-1 * sum(labels * log(softmax(predictions)), dim=-1), dim=0)
    """

    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__(name)
        self.reduction = reduction
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

        if self.reduction == "none":
            return negative_log_loss
        elif self.reduction == "sum":
            reduction_sum = ReduceSum("reduction_sum", negative_log_loss)
            return reduction_sum
        else:
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
