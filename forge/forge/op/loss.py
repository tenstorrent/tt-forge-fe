# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from functools import wraps
from forge.op.tm import Broadcast, Unsqueeze
from ..module import ForgeModule
from .constant import Constant
from .eltwise_unary import Clip, Log, Abs, Sigmoid, Sqrt
from .eltwise_binary import Add, GreaterEqual, Less, Subtract, Multiply
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


class NLLLoss(ForgeModule):
    """
    NLLLoss

    NLLLoss is -1 * sum(labels * log(predictions), dim=-1), optionally reduced using ReduceAvg(default) or ReduceSum.

    Note: This loss expects the input to be log probabilities.
    """

    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__(name)
        self.is_loss = True
        self.reduction = reduction

    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, prediction, labels):
        weighted_prediction = Multiply("mul_pred_labels", prediction, labels)
        negative_one = Constant("negative_one", constant=-1.0)
        loss = Multiply("mul_neg_loss", weighted_prediction, negative_one)
        loss = ReduceSum("r_sum", loss, -1)
        loss = reduce_loss(self.reduction, loss)
        return loss


class KLDivLoss(ForgeModule):
    """
    KLDivLoss

    KLDivLoss is sum(labels * (log(labels) - predictions), dim=-1), optionally reduced using ReduceAvg(default) or ReduceSum.

    Note: This loss expects the input to be log probabilities.
    """

    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__(name)
        self.reduction = reduction
        self.is_loss = True

    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, prediction, labels):
        log_labels = Log("log", labels)
        diff = Subtract("sub", log_labels, prediction)
        product = Multiply("mul", labels, diff)
        loss = reduce_loss(self.reduction, product)
        return loss


class HuberLoss(ForgeModule):
    """
    Huber Loss

    Huber loss is computed as follows:
    loss = if abs(x - y) < delta then 0.5 * (x - y)**2
    loss = if abs(x - y) >= delta then delta * (abs(x - y) - 0.5 * delta)
    """

    def __init__(self, name: str, delta: float = 1.0, reduction: str = "mean"):
        super().__init__(name)
        self.delta = delta
        self.reduction = reduction
        self.is_loss = True

    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, prediction, labels):
        diff = Subtract("sub", prediction, labels)
        abs_diff = Abs("abs", diff)

        # delta
        const_delta = Constant("delta", constant=self.delta)
        # doing manual broadcasting because of the following issue:
        # https://github.com/tenstorrent/tt-metal/issues/15965
        delta = Broadcast("broadcast_delta_1", const_delta, dim=0, shape=abs_diff.shape[0])
        if abs_diff.ndim() != 1:
            delta = Unsqueeze("cast_delta", delta, dim=1)
            delta = Broadcast("broadcast_delta_2", delta, dim=1, shape=abs_diff.shape[1])
        lt_delta = Less("lt_delta", abs_diff, delta)
        ge_delta = GreaterEqual("ge_delta", abs_diff, delta)

        # 0.5 * (x - y)**2
        square = Multiply("square", diff, diff)
        half_const = Constant("half", constant=0.5)
        half = Broadcast("broad_cast_half_dim0", half_const, dim=0, shape=square.shape[0])
        if square.ndim() != 1:
            half = Unsqueeze("half_unsqueeze", half, dim=1)
            half = Broadcast("broad_cast_half_dim1", half, dim=1, shape=square.shape[1])
        half_square = Multiply("half_square", half, square)

        # delta * (abs_diff - 0.5 * delta)
        half_delta = Multiply("half_delta", half, delta)
        delta_diff = Subtract("delta_diff", abs_diff, half_delta)
        mul_delta_diff = Multiply("mul_delta_diff", delta, delta_diff)

        # mask the loss
        # if abs_diff < delta, do 0.5 * (x - y)**2
        # else do delta * (abs_diff - 0.5 * delta)
        loss_lt = Multiply("loss_lt", lt_delta, half_square)
        loss_ge = Multiply("loss_ge", ge_delta, mul_delta_diff)

        # combine masks to get the final loss
        loss = Add("loss", loss_lt, loss_ge)
        return reduce_loss(self.reduction, loss)


def align_shape(target, reference, name):
    unsqueezes = 0
    while target.ndim() < reference.ndim():
        target = Unsqueeze(f"unsqueeze_{name}_{unsqueezes}", target, unsqueezes)
        unsqueezes += 1
    for i in range(target.ndim()):
        if target.shape[i] != reference.shape[i]:
            target = Broadcast(f"broadcast_{name}_{i}", target, i, reference.shape[i])
    return target


class BCELoss(ForgeModule):
    """
    Binary Cross-Entropy Loss

    loss = reduce(-1 * (labels * log(predictions) + (1 - labels) * log(1 - predictions)), dim=0)
    """

    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__(name)
        self.reduction = reduction
        self.is_loss = True

    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, prediction, labels):
        # BCE: -1 * (y * log(p) + (1 - y) * log(1 - p))
        # First term: y * log(p)
        log_prediction = Log("log", prediction)
        first_term = Multiply("mul_lab_pred", labels, log_prediction)

        one = Constant("one", constant=1.0)
        one = align_shape(one, labels, "one")

        # Second term: (1 - y) * log(1 - p)
        one_minus_labels = Subtract("one_minus_labels", one, labels)
        one_minus_prediction = Subtract("one_minus_prediction", one, prediction)
        log_one_minus_prediction = Log("log_one_minus_prediction", one_minus_prediction)
        second_term = Multiply("second_term", one_minus_labels, log_one_minus_prediction)

        # -1 * (y * log(p) + (1 - y) * log(1 - p))
        sum_terms = Add("sum_terms", first_term, second_term)
        neg_one = Constant("neg_one", constant=-1.0)
        neg_one = align_shape(neg_one, sum_terms, "neg_one")
        negative_sum_terms = Multiply("negative_sum_terms", sum_terms, neg_one)
        loss = reduce_loss(self.reduction, negative_sum_terms)
        return loss


class BCEWithLogitsLoss(ForgeModule):
    """
    Binary Cross-Entropy Loss with Logits

    loss = BCELoss(Sigmoid(predictions), labels)
    """

    def __init__(self, name: str, reduction: str = "mean"):
        super().__init__(name)
        self.reduction = reduction
        self.is_loss = True
        self.bce_loss = BCELoss("bce_loss", reduction=self.reduction)

    @validate_shapes(min_dim=1, max_dim=2)
    def forward(self, prediction, labels):
        sigmoid_prediction = Sigmoid("sigmoid", prediction)
        loss = self.bce_loss(sigmoid_prediction, labels)
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
