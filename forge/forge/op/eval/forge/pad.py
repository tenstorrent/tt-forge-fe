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
        # padding is in format [left, right, top, bottom]
        self.padding = padding
        self.mode = mode
        self.value = value
        self.channel_last = channel_last

        return self

    def eval(self, tensors):
        mode_options = ["constant", "replicate", "reflect"]

        # we need to do some permutation of the input tensor
        # since pytorch pad works with channel_last=False
        if self.channel_last:
            # Get the input tensor and its number of dimensions
            input_tensor = tensors[0]
            ndim = len(input_tensor.shape)

            # When channel_last=True, the input tensor is already in format (N, D1, D2, ..., Dn, C)
            # We need to move the channel from the last position to position 1
            # to get (N, C, D1, D2, ..., Dn) which is what PyTorch expects for padding

            # Create permutation that moves channel from the last dim to position 1
            # For a tensor (N, D1, D2, ..., Dn, C) -> (N, C, D1, D2, ..., Dn)
            perm = list(range(ndim))
            perm.insert(1, perm.pop(-1))  # Move last dim (C) to position 1

            # Apply permutation
            transposed = input_tensor.permute(*perm)

            # Apply padding
            padded = torch.nn.functional.pad(transposed, self.padding, mode=mode_options[self.mode], value=self.value)

            # Create reverse permutation to move channel back to the end
            # For a tensor (N, C, D1, D2, ..., Dn) -> (N, D1, D2, ..., Dn, C)
            reverse_perm = list(range(ndim))
            reverse_perm.append(reverse_perm.pop(1))  # Move dim 1 (C) to the end

            # Apply reverse permutation
            return padded.permute(*reverse_perm)
        else:
            return torch.nn.functional.pad(tensors[0], self.padding, mode=mode_options[self.mode], value=self.value)

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, f"Expected 1 input, got {len(tensor_shapes)}"
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
        else:
            total_padding_c = self.padding[0] + self.padding[1]
            total_padding_r = self.padding[2] + self.padding[3]

        # Lower into concats
        left, right, top, bottom = 0, 0, 0, 0
        if len(self.padding) == 2:
            left, right = self.padding[0], self.padding[1]
        elif len(self.padding) == 4:
            left, right, top, bottom = self.padding[0], self.padding[1], self.padding[2], self.padding[3]

        ###############################################################
        if self.mode == 0:  # 'constant' mode
            # TODO: MLIR has ttir::PadOp that works only for constant mode
            # so we could potentialy map it directly to mlir's PadOp
            result = activations
            data_format = activations.output_df

            left_pad, right_pad, top_pad, bot_pad = None, None, None, None

            width_shape = list(result.shape)
            if left > 0:
                width_shape[c_dim_axis] = left
                left_pad = create_pad(dc, width_shape, self.value, data_format)
            if right > 0:
                width_shape[c_dim_axis] = right
                right_pad = create_pad(dc, width_shape, self.value, data_format)

            result = concat_patches(dc, left_pad, result, right_pad, c_dim_axis)

            height_shape = list(result.shape)
            if top > 0:
                height_shape[r_dim_axis] = top
                top_pad = create_pad(dc, height_shape, self.value, data_format)
            if bottom > 0:
                height_shape[r_dim_axis] = bottom
                bot_pad = create_pad(dc, height_shape, self.value, data_format)

            result = concat_patches(dc, top_pad, result, bot_pad, r_dim_axis)

            dc.fuse(result)

            return

        ###############################################################
        if self.mode == 1:  # 'replicate' mode
            result = activations

            left_pad, right_pad, top_pad, bot_pad = None, None, None, None

            if left > 0:
                # extract vector
                left_patch = extract(dc, result, c_dim_axis, 0, 1)
                left_pad = repeat_vector(dc, left_patch, left, c_dim_axis)
            if right > 0:
                # extract vector
                right_patch = extract(dc, result, c_dim_axis, c - 1, c)
                right_pad = repeat_vector(dc, right_patch, right, c_dim_axis)

            result = concat_patches(dc, left_pad, result, right_pad, c_dim_axis)

            if top > 0:
                # extract vector
                top_patch = extract(dc, result, r_dim_axis, 0, 1)
                top_pad = repeat_vector(dc, top_patch, top, r_dim_axis)
            if bottom > 0:
                # extract vector
                bot_patch = extract(dc, result, r_dim_axis, r - 1, r)
                bot_pad = repeat_vector(dc, bot_patch, bottom, r_dim_axis)

            result = concat_patches(dc, top_pad, result, bot_pad, r_dim_axis)

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
    patch = extract(dc, input, dim_axis, start, stop)

    # Mirror patch
    indices = torch.arange(stop - start - 1, -1, -1)
    indices_tensor = dc.tensor(indices, DataFormat.Int32)
    patch_mirrored = dc.op("adv_index", [patch, indices_tensor], (dim_axis,))

    return patch_mirrored


def extract(dc, input, dim_axis, start, stop):
    return dc.op_with_named_attrs(
        "index",
        [input],
        {"dim": dim_axis, "start": start, "stop": stop, "stride": 1},
        (dim_axis, start, stop, 1),
    )


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
    shape = list(shape)
    torch_tensor = torch.full(shape, value, dtype=torch_dtype)

    forge_tensor = dc.tensor(torch_tensor, data_format)
    return forge_tensor


def repeat_vector(dc, input, n_repeats, axis):
    # all axis should have 1 repeat except the axis
    repeats = [1] * len(input.shape)
    repeats[axis] = n_repeats
    repeats = tuple(repeats)
    return dc.op_with_named_attrs("repeat", [input], {"repeats": repeats}, repeats)
