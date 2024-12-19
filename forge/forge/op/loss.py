# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from functools import wraps

from forge.op.tm import Broadcast, Unsqueeze
from ..module import ForgeModule
from .constant import Constant
from .eltwise_unary import Clip, Log, Abs, Sqrt
from .eltwise_binary import Add, Subtract, Multiply
from .nn import Softmax
from .reduce import ReduceSum, ReduceAvg


def validate_shapes(min_dim=None, max_dim=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, predictions, labels, *args, **kwargs):
            assert (
                predictions.ndim() == labels.ndim()
            ), f"Number of dimensions for predictions and labels must match. Got {predictions.ndim()} and {labels.ndim()}."
            if min_dim is not None:
                assert (
                    predictions.ndim() >= min_dim
                ), f"Number of dimensions of predictions and labels must be at least {min_dim}. Got {predictions.ndim()}."
            if max_dim is not None:
                assert (
                    predictions.ndim() <= max_dim
                ), f"Number of dimensions of predictions and labels must be at most {max_dim}. Got {predictions.ndim()}."
            assert (
                predictions.shape == labels.shape
            ), f"Shapes of predictions and labels must match. predictions.shape={predictions.shape} labels.shape={labels.shape}."
            return func(self, predictions, labels, *args, **kwargs)

        return wrapper

    return decorator


def reduce_loss(reduction, loss):
    if reduction == "none":
        return loss

    reduction_op = {
        "mean": ReduceAvg,
        "sum": ReduceSum,
    }

    if reduction not in reduction_op:
        raise RuntimeError("Unsupported reduce type: " + reduction)

    op = reduction_op[reduction]
    dims = loss.ndim()
    # hack for 1D tensors described in issue
    # https://github.com/tenstorrent/tt-forge-fe/issues/907
    for i in range(-1, -1 - dims, -1):
        loss = op(f"reduce_loss_{reduction}_dim_{i}", loss, i)
    return loss


class CrossEntropyLoss(ForgeModule):
    """
    Cross-Entropy Loss (with mean reduction)

    loss = reduce_avg(-1 * sum(labels * log(softmax(predictions)), dim=-1), dim=0)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.is_loss = True

    @validate_shapes(min_dim=2, max_dim=2)
    def forward(self, predictions, labels):
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

        reduction_mean = ReduceAvg("reduction_mean", negative_log_loss, dim=0)
        return reduction_mean


class L1Loss(ForgeModule):
    """
    L1Loss

    L1Loss is abs(labels - prediction), optionally reduced using ReduceAvg(default) or ReduceSum.
    """

    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__(name)
        self.reduction = reduction
        self.is_loss = True

    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, prediction, labels):
        diff = Abs("abs", Subtract("sub", prediction, labels))
        loss = reduce_loss(self.reduction, diff)
        return loss


class MSELoss(ForgeModule):
    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__(name)
        self.reduction = reduction
        self.is_loss = True

    # ndim > 2 does not work all the time because of the following issue:
    # https://github.com/tenstorrent/tt-metal/issues/15996
    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, predictions, labels):
        diff = Subtract("sub", predictions, labels)
        square = Multiply("square", diff, diff)
        loss = reduce_loss(self.reduction, square)
        return loss


class TripletMarginLoss(ForgeModule):
    def __init__(self, name: str, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(name)
        self.margin = margin
        # pow is fixed to 2.0
        # https://github.com/tenstorrent/tt-mlir/issues/1203
        self.p = 2.0
        self.reduction = reduction
        self.is_loss = True

    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, anchor, positive, negative):
        if anchor.ndim() == 1:
            anchor = Unsqueeze("unsqueeze_anchor", anchor, 0)
            positive = Unsqueeze("unsqueeze_positive", positive, 0)
            negative = Unsqueeze("unsqueeze_negative", negative, 0)

        # Squared distance for positive pair
        sub_pos = Subtract("sub_pos", anchor, positive)
        square_pos = Multiply("square_pos", sub_pos, sub_pos)
        square_pos = ReduceSum("reduce_pos_sum_dim_1", square_pos, 1)
        pos_dist = Sqrt("sqrt_pos", square_pos)

        # Squared distance for negative pair
        sub_neg = Subtract("sub_neg", anchor, negative)
        square_neg = Multiply("square_neg", sub_neg, sub_neg)
        square_neg = ReduceSum("reduce_neg_sum_dim_1", square_neg, 1)
        neg_dist = Sqrt("sqrt_neg", square_neg)

        # Compute triplet loss
        dist_diff = Subtract("dist_diff", pos_dist, neg_dist)

        # Margin
        # doing manual broadcasting because of the following issue:
        # https://github.com/tenstorrent/tt-metal/issues/15965
        margin = Constant("margin", constant=self.margin)
        margin = Unsqueeze("expand_margin", margin, -1)
        margin = Broadcast("broadcast_margin", margin, 0, dist_diff.shape[0])
        margin = Broadcast("broadcast_margin_dim_1", margin, 1, dist_diff.shape[1])

        dist_with_margin = Add("dist_with_margin", dist_diff, margin)

        # clip to (0, inf)
        clip_dist = Clip("clip_dist", dist_with_margin, 0.0, float("inf"))
        loss = reduce_loss(self.reduction, clip_dist)
        return loss
