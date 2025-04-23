# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional
from ..interface import PyTM
from forge._C import DataFormat
from forge.tensor import forge_dataformat_to_pytorch_dtype


class Pad(PyTM):
    @classmethod
    def create(cls, padding, mode, value, channel_last):
        self = cls("pad")
        self.padding = padding
        self.mode = mode
        self.channel_last = channel_last
        self.value = value
        return self

    def eval(self, tensors):
        assert len(self.padding) == 2 or len(self.padding) == 4

        mode_options = ["constant", "replicate", "reflect"]
        return torch.nn.functional.pad(tensors[0], self.padding, mode=mode_options[self.mode], value=self.value)

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1
        shape = list(tensor_shapes[0])

        if self.channel_last:
            shape[-2] += self.padding[0] + self.padding[1]  # width padding
            if len(self.padding) == 4:
                shape[-3] += self.padding[2] + self.padding[3]  # height padding
        else:
            shape[-1] += self.padding[0] + self.padding[1]  # width padding
            if len(self.padding) == 4:
                shape[-2] += self.padding[2] + self.padding[3]  # height padding
        return tuple(shape), []

    def decompose(self, dc, inputs):
        if all([x == 0 for x in self.padding]):
            # Pad size is 0
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)
            return

        activations = inputs[0]
        if self.channel_last:
            r = activations.shape[-3]
            r_dim_axis = -3
            c = activations.shape[-2]
            c_dim_axis = -2
        else:
            r = activations.shape[-2]
            r_dim_axis = -2
            c = activations.shape[-1]
            c_dim_axis = -1

        # convert axis to positive numbers due to the decomposition ops working with positive numbers
        r_dim_axis = r_dim_axis if r_dim_axis >= 0 else len(activations.shape) + r_dim_axis
        c_dim_axis = c_dim_axis if c_dim_axis >= 0 else len(activations.shape) + c_dim_axis

        # R, C are flipped because pytorch pad starts from last axis
        if len(self.padding) == 2:
            total_padding_c = self.padding[0] + self.padding[1]
            total_padding_r = 0
        elif len(self.padding) == 4:
            total_padding_c = self.padding[0] + self.padding[1]
            total_padding_r = self.padding[2] + self.padding[3]
        else:
            raise RuntimeError("Forge only support Pad with either 2 or 4 padding tuple size")

        # Lower into concats
        left, right, top, bottom = 0, 0, 0, 0
        if len(self.padding) == 2:
            left, right = self.padding
        elif len(self.padding) == 4:
            left, right, top, bottom = self.padding

        ###############################################################
        if self.mode == 0:  # 'constant' mode
            # mlir only supports padding on the last 2 dimensions, so we need to transpose the tensor
            if self.channel_last:
                result = activations

                left_pad, right_pad, top_pad, bot_pad = None, None, None, None
                height_dim, width_dim = -2 - int(self.channel_last), -1 - int(self.channel_last)

                width_shape = [1] * len(result.shape)
                if left > 0:
                    width_shape[width_dim] = left
                    left_pad = create_pad(dc, width_shape, self.value, result.data_format)
                if right > 0:
                    width_shape[width_dim] = right
                    right_pad = create_pad(dc, width_shape, self.value, result.data_format)

                result = concat_patches(dc, left_pad, result, right_pad, width_dim)

                height_shape = [1] * len(result.shape)
                if top > 0:
                    height_shape[height_dim] = top
                    top_pad = create_pad(dc, height_shape, self.value, result.data_format)
                if bottom > 0:
                    height_shape[height_dim] = bottom
                    bot_pad = create_pad(dc, height_shape, self.value, result.data_format)

                result = concat_patches(dc, top_pad, result, bot_pad, height_dim)

                dc.fuse(result)
            return

        ###############################################################
        if self.mode == 1:  # 'replicate' mode
            result = activations

            if self.channel_last:
                result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])

                orig_shape = result.shape
                result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]))
                result = dc.op(TransposeTM.create(-2, -1), [result])
                spm = create_pad_replicate_sparse_picker(c, r, top, bottom, left, right)
                spm = dc.tensor(spm)
                result = dc.op("sparse_matmul", [spm, result])
                result = dc.op(TransposeTM.create(-2, -1), [result])
                result = dc.op(
                    "reshape",
                    [result],
                    (1, orig_shape[-3], orig_shape[-1] + total_padding_r, orig_shape[-2] + total_padding_c),
                )

                result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
            else:
                orig_shape = result.shape
                if len(orig_shape) == 2:
                    result = dc.op("reshape", [result], (1, orig_shape[-2] * orig_shape[-1]))
                else:
                    result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]))
                result = dc.op(TransposeTM.create(-2, -1), [result])
                spm = create_pad_replicate_sparse_picker(r, c, left, right, top, bottom)
                spm = dc.tensor(spm)
                result = dc.op("sparse_matmul", [spm, result])
                result = dc.op(TransposeTM.create(-2, -1), [result])
                if len(orig_shape) == 2:
                    result = dc.op(
                        "reshape", [result], (orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)
                    )
                else:
                    result = dc.op(
                        "reshape",
                        [result],
                        (1, orig_shape[-3], orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c),
                    )

            dc.fuse(result)
            return

        ###############################################################
        elif self.mode == 2:  # Reflect mode
            result = activations

            if left > c - 1 or right > c - 1:
                raise RuntimeError(f"Both left padding ({left}) and right padding ({right}) has to be max {c - 1} each")

            if top > r - 1 or bottom > r - 1:
                raise RuntimeError(f"Both top padding ({top}) and bottom padding ({bottom}) has to be max {r - 1} each")

            # Step 1: Extract left and right patches which are on the c axis (width) and mirror them horizontally
            left_patch_mirrored, right_patch_mirrored = None, None
            if left > 0:
                left_patch_mirrored = extract_and_mirror(dc, result, c_dim_axis, 1, left + 1)

            if right > 0:
                right_patch_mirrored = extract_and_mirror(dc, result, c_dim_axis, c - right - 1, c - 1)

            # Step 2: Concatenate the mirrored patches to the original result
            result = concat_patches(dc, left_patch_mirrored, result, right_patch_mirrored, c_dim_axis)

            # Step 3: Extract top and bottom patches which are on the r axis (height) and mirror them vertically
            top_patch_mirrored, bot_patch_mirrored = None, None
            if top > 0:
                top_patch_mirrored = extract_and_mirror(dc, result, r_dim_axis, 1, top + 1)

            if bottom > 0:
                bot_patch_mirrored = extract_and_mirror(dc, result, r_dim_axis, r - bottom - 1, r - 1)

            # Step 4: Concatenate the mirrored patches to the original result
            result = concat_patches(dc, top_patch_mirrored, result, bot_patch_mirrored, r_dim_axis)

            dc.fuse(result)
            return
            dc.fuse(result)
            return

    def backward(self, ac, operand, inputs, output, grad):
        # TODO: Check whether this is valid backward
        assert len(self.padding) == 2 or len(self.padding) == 4, "Not supported padding type"

        height_dim, width_dim = -2 - int(self.channel_last), -1 - int(self.channel_last)
        original_height, original_width = grad.shape[height_dim], grad.shape[width_dim]

        if len(self.padding) == 4:
            pad_left, pad_right, pad_top, pad_bottom = self.padding
            grad = ac.op(
                "narrow", (grad,), (height_dim, pad_top, original_height - pad_top - pad_bottom, original_height)
            )
            return ac.op(
                "narrow", (grad,), (width_dim, pad_left, original_width - pad_left - pad_right, original_width)
            )
        else:
            pad_left, pad_right = self.padding
            return ac.op(
                "narrow", (grad,), (width_dim, pad_left, original_width - pad_left - pad_right, original_width)
            )


def extract_and_mirror(dc, input, dim_axis, start, stop):
    # Extract patch
    patch = dc.op_with_named_attrs(
        "index",
        [input],
        {"dim": dim_axis, "start": start, "stop": stop, "stride": 1},
        (dim_axis, start, stop, 1),
    )

    # Mirror patch
    indices = torch.arange(stop - start - 1, -1, -1)
    indices_tensor = dc.tensor(indices, DataFormat.Int32)
    patch_mirrored = dc.op("adv_index", [patch, indices_tensor], (dim_axis,))

    return patch_mirrored


def concat_patches(dc, first_patch, center, second_patch, dim_axis):
    if first_patch and second_patch:
        return dc.op_with_named_attrs(
            "concatenate", [first_patch, center, second_patch], {"dim": dim_axis}, (dim_axis,)
        )
    elif first_patch:
        return dc.op_with_named_attrs("concatenate", [first_patch, center], {"dim": dim_axis}, (dim_axis,))
    elif second_patch:
        return dc.op_with_named_attrs("concatenate", [center, second_patch], {"dim": dim_axis}, (dim_axis,))
    else:
        return center


def create_pad(dc, shape, value, data_format):
    torch_dtype = forge_dataformat_to_pytorch_dtype(data_format)
    torch_tensor = torch.full(shape, value, dtype=torch_dtype)

    forge_tensor = dc.tensor(torch_tensor, data_format)
    return forge_tensor
