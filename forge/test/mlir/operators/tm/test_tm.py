# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.tensor import to_forge_tensors
from forge.verify.verify import verify

# @pytest.mark.xfail(reason="RuntimeError: Input must be UINT32 or BFLOAT16")
@pytest.mark.parametrize(
    "input_shape, sequence_lengths",
    [
        ((1, 32, 2), [5]),
        ((1, 64, 4), [55]),
        ((1, 16, 8), [14]),
        ((1, 164, 14), [22]),
        ((1, 80, 7), [79]),
        ((1, 43, 25), [34]),
    ],
)
def test_multi_indexing(input_shape, sequence_lengths):
    class Multi_Indexing(torch.nn.Module):
        def __init__(self, sequence_lengths):
            super().__init__()
            self.sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.int64)

        def forward(self, logits):
            pooled_logits = logits[torch.arange(1), self.sequence_lengths]
            return pooled_logits

    inputs = [torch.randn(input_shape)]
    framework_model = Multi_Indexing(sequence_lengths)
    framework_model.eval()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape,dim,index",
    [
        ((3, 23, 73, 164), 1, 21),
        ((8, 66, 713, 54), 2, -403),
        ((12, 86, 273, 34), 3, 30),
        ((5, 115, 75, 64), -3, -21),
        ((2, 7, 213, 64), -2, -103),
        ((6, 99, 12, 64), -1, 36),
        ((1, 6, 73, 64), 2, -2),
        ((1, 6, 73, 64), -2, -1),
        ((3, 27, 94), 0, 2),
        ((5, 100, 64), 1, 50),
        ((7, 82, 16), 2, -5),
        ((10, 53, 23), -1, 15),
        ((8, 32, 12), -2, -20),
        ((18, 31, 22), -3, -12),
    ],
)
@pytest.mark.push
def test_index(shape, dim, index):
    class Index(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, x):

            if dim == 3 or dim == -1:
                return x[..., [self.index]]

            elif dim == 2 or dim == -2:
                return x[..., [self.index], :]

            elif dim == 1 or dim == -3:
                return x[:, [self.index], ...]

            else:
                return x[[self.index], ...]

    inputs = [torch.rand(shape)]

    framework_model = Index(index)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 256, 6, 6),
        (1, 3, 64, 64),
        (1, 512, 14, 14),
        (1, 3, 224, 224),
        (2, 256, 10, 10),
        (1, 512, 3, 3),
        (1, 1000, 1, 1),
        (2, 128, 8, 8),
        (4, 1, 32, 32),
        (8, 64, 32, 32),
    ],
)
@pytest.mark.push
def test_flatten(shape):
    class Flatten(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.flatten(x, 1)

    inputs = [torch.rand(shape)]

    framework_model = Flatten()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("operand_and_cast_dtype", [(torch.float32, torch.int32), (torch.int32, torch.float32)])
@pytest.mark.push
def test_cast(operand_and_cast_dtype):

    operand_dtype = operand_and_cast_dtype[0]
    cast_dtype = operand_and_cast_dtype[1]

    class Cast(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return a.to(cast_dtype)

    def get_input_tensor(dtype):
        shape = (1, 32, 32)
        if dtype in [torch.float32, torch.bfloat16]:
            return torch.rand(shape, dtype=dtype)
        elif dtype in [torch.int32]:
            return torch.randint(high=torch.iinfo(dtype).max, size=shape, dtype=dtype)
        else:
            raise Exception("Unsupported datatype")

    inputs = [
        get_input_tensor(operand_dtype),
    ]

    framework_model = Cast()
    framework_model.eval()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "batch_size, num_channels, height, width",
    [
        (1, 32, 56, 56),
    ],
)
@pytest.mark.push
def test_layernorm(batch_size, num_channels, height, width):

    # framework_model = nn.LayerNorm((num_channels, height, width)) # Support only normalization over last one dimension
    framework_model = nn.LayerNorm((width))

    inputs = [torch.rand(batch_size, num_channels, height, width)]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


params = [
    ((1, 32, 64), (-1, -2)),
    ((1, 64, 32), (1, 2)),
    ((1, 32, 64, 128), (3, 2)),
    ((32, 128), (0, 1)),
    ((18, 65), (1, 0)),
    ((6, 33, 34), (-1, 1)),
    ((1, 32, 64), (-2, -3)),
    ((6, 33, 34), (-1, -3)),
    ((32, 128, 24), (1, -3)),
    ((1, 12, 32, 100), (-3, -2)),
    ((32, 12, 100), (-1, -2)),
]
# Dynamically generate params with conditional xfail
param_list = []
for param in params:
    for data_format in [torch.float32, torch.bfloat16]:
        param_list.append((param, data_format))


@pytest.mark.parametrize("params, data_format", param_list)
@pytest.mark.push
def test_transpose(params, data_format):
    class Transpose(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, a):
            return torch.transpose(a, *self.dims)

    input_shape, dims = params
    inputs = [torch.rand(input_shape, dtype=data_format)]  # Use data_format instead of hardcoded dtype
    # Initialize the model with data_formats
    framework_model = Transpose(dims).to(data_format)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "source_and_target_shape",
    [((8, 32, 256), (2, 4, 32, 256)), ((8, 32, 32), (1, 2, 4, 32, 32)), ((8192, 128), (1, 256, 32, 128))],
    ids=["1", "2", "3"],
)
@pytest.mark.push
def test_reshape(source_and_target_shape):
    source_shape, target_shape = source_and_target_shape

    class Reshape(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.reshape(a, target_shape)

    inputs = [torch.rand(source_shape)]

    framework_model = Reshape()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((1, 8, 16, 32, 32), 0),
        ((8, 1, 16, 32, 32), 1),
        ((8, 16, 1, 32, 32), 2),
        ((1, 8, 16, 32, 32), -5),
        ((8, 1, 16, 32, 32), -4),
        ((8, 16, 1, 32, 32), -3),
        ([1, 12, 3200], 0),
        ([1, 1, 2048, 1], [-3, -4]),
        ([1, 64, 1, 1], [-1, -4]),
        ([1, 1, 1, 128], [-2, -4]),
        ([1, 1, 32, 1], [-1, -3]),
        ([1, 1, 1, 64], [-4, -3]),
    ],
)
@pytest.mark.push
def test_squeeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    if input_shape == [1, 12, 3200] or isinstance(dim, list) and len(dim) > 1 and all(d < 0 for d in dim):
        pytest.xfail("TTNN: Tensor layout issues with non tile dim aligned shapes")

    class Squeeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.squeeze(a, dim)

    inputs = [torch.rand(*input_shape)]

    framework_model = Squeeze()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((8, 16, 32, 32), 0),
        ((8, 16, 32, 32), 1),
        ((8, 16, 32, 32), 2),
        ((8, 16, 32, 32), -3),
        ((8, 16, 32, 32), -4),
        ([12, 8640], 0),
    ],
)
@pytest.mark.push
def test_unsqueeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    if input_shape == [12, 8640]:
        pytest.xfail("TTNN: Tensor layout issues with non tile dim aligned shapes")

    class Unsqueeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.unsqueeze(a, dim)

    inputs = [torch.rand(*input_shape)]

    framework_model = Unsqueeze()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("dim", [-1, -2, -3], ids=["-1", "-2", "-3"])
@pytest.mark.parametrize("start", [0], ids=["0"])
@pytest.mark.parametrize("stop", [2, 32, 64], ids=["2", "32", "64"])
@pytest.mark.parametrize("stride", [1, 2, 4, 8], ids=["1", "2", "4", "8"])
@pytest.mark.parametrize("shape", [(1, 32, 64, 64), (32, 64, 64), (64, 64)])
@pytest.mark.push
def test_indexing(dim, start, stop, stride, shape):
    if len(shape) == 2 and dim == -3:
        pytest.skip("Skipping since indexing on dim=-3, 2D tensor doesn't make sense")
    if stop > shape[dim]:
        pytest.skip("Skipping since stop > shape[dim]")

    class ForgeIndexing(forge.ForgeModule):
        def __init__(self, dim, start, stop, stride):
            super().__init__("ForgeTest")

        def forward(self, x):
            return forge.op.Index("indexing_op_1", x, dim, start, stop, stride)

    inputs = to_forge_tensors([torch.rand(*shape)])

    framework_model = ForgeIndexing(dim, start, stop, stride)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "indices_shape",
    [
        (12,),
        (32,),
        (1, 7),
        (1, 28),
    ],
)
@pytest.mark.parametrize(
    "input_tensor_shape",
    [
        (12, 100),
        (3200, 512),
        (2048, 128),
        (4127, 256),
    ],
)
@pytest.mark.push
def test_adv_index_embedding_decompostion(indices_shape, input_tensor_shape):
    class ForgeAdvIndex(forge.ForgeModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, input_tensor, indices):
            return forge.op.AdvIndex("adv_index_op_1", input_tensor, indices)

    framework_model = ForgeAdvIndex("ForgeAdvIndex")

    # Sample Inputs
    pt_input_tensor = torch.rand(input_tensor_shape).to(torch.float32)
    pt_indices = torch.randint(input_tensor_shape[0], indices_shape).to(torch.int32)
    inputs = to_forge_tensors([pt_input_tensor, pt_indices])

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_reshape_pytorch():
    class ReshapeTest(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp_1, inp_2):
            inp = inp_1 + inp_2
            inp_res = inp.reshape(1, 2, 2, 7, 7, 384)
            inp_res = inp_res.transpose(-4, -3)
            inp_res = inp_res.reshape(-1, 384)
            return inp_res

    inputs = [torch.rand(4, 49, 384), torch.rand(4, 49, 384)]

    framework_model = ReshapeTest()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_broadcast_pytorch():
    class BroadcastTest(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp_1):
            inp_1 = inp_1.transpose(-3, -2)
            inp_1_1 = inp_1[:1]
            inp_1_1 = inp_1_1.squeeze(0)
            return inp_1_1

    inputs = [torch.rand(3, 64, 49, 3, 32)]

    framework_model = BroadcastTest()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    ["input_shapes", "dim"],
    [
        pytest.param(
            [(1, 256, 24, 24), (1, 256, 24, 24)],
            -4,
        ),
        pytest.param(
            [(5, 64, 128, 128), (5, 64, 128, 128)],
            -3,
        ),
        pytest.param(
            [(1, 30, 30, 16), (1, 30, 30, 16)],
            -2,
            marks=pytest.mark.xfail(reason="Trying to access element outside of dimensions: 4"),
        ),
        pytest.param(
            [(1, 30, 30, 16), (1, 30, 30, 16)],
            3,
            marks=pytest.mark.xfail(reason="Trying to access element outside of dimensions: 4"),
        ),
        pytest.param(
            [(5, 64, 128, 128), (5, 64, 128, 128)],
            -1,
            marks=pytest.mark.xfail(
                reason="Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 10584992 B which is beyond max L1 size of 1499136 B"
            ),
        ),
        pytest.param(
            [(1, 256, 24, 24), (1, 256, 24, 24)],
            4,
        ),
        pytest.param(
            [(1, 256, 24, 24), (1, 256, 24, 24)],
            2,
        ),
        pytest.param(
            [(5, 64, 128, 128), (5, 64, 128, 128)],
            1,
        ),
        pytest.param(
            [(1, 30, 30, 16), (1, 30, 30, 16)],
            0,
        ),
    ],
)
def test_stack(input_shapes, dim):
    class Stack(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, *tensors):
            return torch.stack(tensors, dim=self.dim)

    inputs = [torch.rand(shape) for shape in input_shapes]

    framework_model = Stack(dim)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name="stack_sanity")
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - repeat"
)
@pytest.mark.push
def test_repeat():
    class Repeat(nn.Module):
        def __init__(self, repeats):
            super().__init__()
            self.repeats = repeats

        def forward(self, x):
            return x.repeat(*self.repeats)

    inputs = [torch.rand(1, 2, 1, 4, 4)]

    framework_model = Repeat(repeats=(1, 1, 4, 1, 1))
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - repeat_interleave"
)
@pytest.mark.push
def test_expand():
    class Expand(nn.Module):
        def __init__(self, expand_shape):
            super().__init__()
            self.expand_shape = expand_shape

        def forward(self, x):
            return x.expand(*self.expand_shape)

    inputs = [torch.rand(1, 2, 1, 4, 4)]

    framework_model = Expand(expand_shape=(1, 2, 4, 4, 4))
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - repeat_interleave"
)
@pytest.mark.push
def test_repeat_interleave():
    class RepeatInterleave(nn.Module):
        def __init__(self, repeats, dim):
            super().__init__()
            self.repeats = repeats
            self.dim = dim

        def forward(self, x):
            return x.repeat_interleave(self.repeats, dim=self.dim)

    inputs = [torch.rand(1, 2, 1, 4, 4)]

    framework_model = RepeatInterleave(repeats=4, dim=2)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("shape", [(1, 32, 64, 64), (32, 64, 64), (64, 64)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("begin", [0, 16])
@pytest.mark.parametrize("length", [4, 16])
@pytest.mark.parametrize("stride", [16, 32])
def test_select(shape, dim, begin, length, stride):
    if stride <= begin + length:
        pytest.skip("Skipping since stride <= begin + length")

    class Select(forge.ForgeModule):
        def __init__(self):
            super().__init__("Select")

        def forward(self, x):
            x = forge.op.Select("select_op", x, dim, [begin, length], stride)
            return x

    inputs = to_forge_tensors([torch.rand(*shape)])
    framework_model = Select()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)