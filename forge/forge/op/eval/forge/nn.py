# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import numpy as np
import torch
import torch.nn.functional as F

from ..common import to_torch_operands


def eval(op_type, attr, ops):
    """
    Operator or module evaluation function. Evaluation is done using PyTorch ML library.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    ops:
        Operation operands.

    Returns
    -------
        Result of the evaluation.

    """

    if op_type == "batchnorm":

        assert len(ops) == 5, "batchnorm should have five operands."
        assert len(attr) == 1, "batchnorm should have one attributes."

        t_ops = to_torch_operands(*ops)

        input_ = t_ops[0]  # Input tensor
        weight = t_ops[1]  # weights, weight re-scaling parameter
        bias = t_ops[2]  # bias, weight re-centering parameter
        running_mean = t_ops[3]
        running_var = t_ops[4]
        epsilon = attr[0]

        # assert gamma.shape[-1] == input_.shape[-1], "Weights shape must be the same as normalized shape."
        # for gdim in gamma.shape[:-1]:
        #    assert gdim == 1, "All dimensions but the last one must be 1"
        # assert beta.shape[-1] == input_.shape[-1], "Bias shape must be the same as normalized shape."
        # for bdim in beta.shape[:-1]:
        #    assert bdim == 1, "All dimensions but the last one must be 1"

        return F.batch_norm(
            input=input_,
            running_mean=running_mean.shape[-1:],
            running_var=running_var.shape[-1:],
            normalized_shape=input_.shape[-1:],
            weight=weight.reshape(gamma.shape[-1:]),
            bias=bias.reshape(beta.shape[-1:]),
            eps=epsilon,
        )

    assert False, f"{op_type} is not defined in nn eval."


def shape(op_type, attr, ops):
    """
    Computes output shapes for particular operation.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    ops:
        Operation operands.

    Returns
    -------
        Output shape after particular operation.

    """

    if op_type == "batchnorm":

        assert len(ops) == 5, "Layernorm should have five operands."
        assert len(attr) == 1, "Layernorm should have one attributes."

        return ops[0], []

    assert False, f"{op_type} is not defined in nn shape."


def backward(op_type, attr, ac, operand, inputs, output, grad):
    """
    Computes backward value for particular operation using derivative
    of the current operation and gradient from the previous node.


    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    ac:
        Autograd Context, Forge C++ API for automatic gradient computation.

    operand:
        Operation operands.

    inputs:
        Operation inputs.

    output:
        Operation outpus, i.e. reuslt of the operation, from forward pass.
        If it's needed in backward it shouldn't be computed again, just re-used.

    grad:
        Gradient of the previous node.

    Returns
    -------
        Result of the backward pass.

    """

    if op_type == "batchnorm":
        raise NotImplementedError("Back propagation for Batchnorm op is not implemented yet")

    assert False, f"{op_type} is not defined in nn backward. "


def decompose(op_type, attr, dc, inputs):
    """
    Decompses the operator after backward pass.

    Parameters
    ----------
    op_type:
        Type of the operation.

    attr:
        Operation attributes.

    dc:
        Decomposing Context, Forge C++ API for breaking
        Forge graph/subgraph to simpler, Forge graph, too.

    inputs:
        Operation inputs.

    Returns
    -------
        Result of the operation.

    """

    if op_type == "batchnorm":
        assert len(inputs) == 5, "Batchnorm should have five operands."
        assert len(attr) == 1, "Layernorm should have one attributes."

        input_ = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        running_mean = inputs[3]
        running_var = inputs[4]
        epsilon = attr[0]

        # const tensor
        eps_tensor = dc.tensor(torch.zeros(running_var.shape.as_list()) + epsilon)
        neg_one = dc.tensor(torch.zeros(running_mean.shape.as_list()) - 1.0)

        # decompose
        var_eps = dc.op("add", (running_var, eps_tensor), ())
        sqrt = dc.op("sqrt", (var_eps,), ())
        recipro = dc.op("reciprocal", (sqrt,), ())
        weighted = dc.op("multiply", (recipro, weight), ())
        neg_mean = dc.op("multiply", (neg_one, running_mean), ())
        weighted_mean = dc.op("multiply", (weighted, neg_mean), ())
        weighted_bias = dc.op("add", (weighted_mean, bias), ())
        weighted_bias = dc.op_with_named_attrs(
            "unsqueeze",
            [weighted_bias],
            {"dim": 1},
        )
        weighted_bias = dc.op_with_named_attrs(
            "unsqueeze",
            [weighted_bias],
            {"dim": 1},
        )
        weighted_var = dc.op_with_named_attrs(
            "unsqueeze",
            [weighted],
            {"dim": 1},
        )
        weighted_var = dc.op_with_named_attrs(
            "unsqueeze",
            [weighted_var],
            {"dim": 1},
        )
        scaled = dc.op("multiply", (input_, weighted_var), ())
        biased = dc.op("add", (scaled, weighted_bias), ())
        dc.fuse(biased)
        return
