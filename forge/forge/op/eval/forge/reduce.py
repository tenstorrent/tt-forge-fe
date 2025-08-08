# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from ..common import to_torch_operands


def eval(type, attr, ops):
    """
    Evaluation function for reduce operations
    """
    assert len(ops) == 1, f"Reduce ops should have one input {len(ops)} {attr}"
    t_ops = to_torch_operands(*ops)

    if type == "argsort":
        axis = attr.get("axis", -1)
        is_ascend = attr.get("is_ascend", True)
        descending = not is_ascend
        return torch.argsort(t_ops[0], dim=axis, descending=descending)

    if type == "sort":
        axis = attr.get("axis", -1)
        is_ascend = attr.get("is_ascend", True)
        descending = not is_ascend
        values, _ = torch.sort(t_ops[0], dim=axis, descending=descending)
        return values

    else:
        raise RuntimeError(f"Reduce operation not implemented: {type}")


def shape(type, attr, ops):
    """
    Shape computation for reduce operations
    """
    assert len(ops) == 1, f"Reduce ops should have one input"
    input_shape = ops[0]

    if type == "argsort":
        # Argsort preserves the input shape
        return list(input_shape)

    else:
        raise RuntimeError(f"Reduce operation shape not implemented: {type}")


def backward(type, attr, ops):
    """
    Backward pass for reduce operations (most don't have gradients)
    """
    if type == "argsort":
        raise RuntimeError("Argsort does not support backward pass")
    else:
        raise RuntimeError(f"Reduce operation backward not implemented: {type}")
