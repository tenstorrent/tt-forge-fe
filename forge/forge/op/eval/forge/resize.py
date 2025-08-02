# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def shape(type, attr, ops):
    if type == "downsample2d":
        channel_last = attr[2]
        scale_factor = attr[0]
        shape = list(ops[0])
        if channel_last:
            shape[-3], shape[-2] = shape[-3] // scale_factor, shape[-2] // scale_factor
        else:
            shape[-2], shape[-1] = shape[-2] // scale_factor, shape[-1] // scale_factor
        return shape, []


def backward(type, attr, ac, operand, inputs, output, grad):
    raise RuntimeError("This should never be called.")
