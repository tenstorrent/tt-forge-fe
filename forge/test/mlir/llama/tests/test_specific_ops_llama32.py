# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from torch import nn

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify, verify_backward


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 11, 1), (1,)),
        ((1, 32, 11, 64), (1, 32, 11, 64)),
        ((1, 8, 11, 64), (1, 8, 11, 64)),
        ((1, 32, 11, 11), (1, 1, 11, 11)),
        ((1, 11, 2048), (1, 11, 2048)),
    ],
)
@pytest.mark.push
def test_add(forge_property_recorder, shapes):
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [torch.rand(shapes[0]), torch.rand(shapes[1])]

    framework_model = Add()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "inputs_and_dim",
    [
        ((1, 11, 32), (1, 11, 32), -1),
        ((1, 32, 11, 32), (1, 32, 11, 32), -1),
        ((1, 8, 11, 32), (1, 8, 11, 32), -1),
    ],
)
@pytest.mark.push
def test_concat(forge_property_recorder, inputs_and_dim):
    in_shape1, in_shape2, dim = inputs_and_dim

    class Concat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.cat((a, b), dim)

    inputs = [torch.rand(in_shape1), torch.rand(in_shape2)]

    framework_model = Concat()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("shapes", [(1, 11, 64)])
@pytest.mark.push
def test_cosine(forge_property_recorder, shapes):
    class Cosine(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.cos(a)

    inputs = [torch.rand(shapes)]

    framework_model = Cosine()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("shapes", [(1, 11, 64)])
@pytest.mark.push
def test_sine(forge_property_recorder, shapes):
    class Sine(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sin(a)

    inputs = [torch.rand(shapes)]

    framework_model = Sine()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 11), 128256, 2048),
    ],
)
@pytest.mark.push
def test_embedding(forge_property_recorder, shapes):
    input_size, vocab_size, embedding_dim = shapes

    class Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

    inputs = [torch.randint(0, vocab_size, input_size)]

    framework_model = Embedding()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# @pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize(
    "shapes, train",
    [
        (((11, 2048), (2048, 2048)), False),
        (((1, 32, 1), (1, 1, 11)), False),
        (((11, 2048), (2048, 512)), False),
        (((32, 11, 64), (32, 64, 11)), False),
        (((32, 11, 11), (32, 11, 64)), False),
        (((11, 2048), (2048, 8192)), False),
        (((1, 11, 8192), (8192, 2048)), False),
        (((1, 11, 2048), (2048, 128256)), False),
        (((32, 11, 64), (32, 64, 11)), True),
    ],
)
@pytest.mark.push
def test_matmul(forge_property_recorder, shapes, train):
    shape1, shape2 = shapes

    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    inputs = [
        torch.rand(shape1, requires_grad=train),
        torch.rand(shape2, requires_grad=train),
    ]

    framework_model = Matmul()
    framework_model.eval() if not train else framework_model.train()

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, training=train, forge_property_handler=forge_property_recorder
    )

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
    )
    if train:
        output_grad = torch.rand_like(fw_out[0])

        verify_backward(
            inputs,
            output_grad,
            fw_out[0],
            co_out[0],
            framework_model,
            compiled_model,
            verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        )


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize(
    "shape, dim, repeats",
    [
        ((1, 8, 1, 12, 64), 0, 1),
        ((1, 8, 1, 12, 64), 2, 4),
    ],
)
@pytest.mark.push
def test_repeat_interleave(forge_property_recorder, shape, dim, repeats, train):
    class RepeatInterleave(nn.Module):
        def __init__(self, dim, repeats):
            super().__init__()
            self.repeats = repeats
            self.dim = dim

        def forward(
            self,
            x,
        ):
            return torch.repeat_interleave(x, repeats=repeats, dim=dim)

    inputs = [torch.rand(shape, requires_grad=train)]

    framework_model = RepeatInterleave(dim=dim, repeats=repeats)

    framework_model.eval() if not train else framework_model.train()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, training=train, forge_property_handler=forge_property_recorder)

    fw_out, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    if train:
        output_grad = torch.rand_like(fw_out[0])
        verify_backward(inputs, output_grad, fw_out[0], co_out[0], framework_model, compiled_model)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 11, 2048), (1, 11, 2048)),
        ((1, 11, 2048), (1, 11, 1)),
        ((2048,), (1, 11, 2048)),
        ((1, 32, 11, 64), (1, 1, 11, 64)),
        ((1, 32, 11, 32), (1,)),
        ((1, 8, 11, 64), (1, 1, 11, 64)),
        ((1, 8, 11, 32), (1,)),
        ((1, 32, 11, 11), (1,)),
        ((1, 11, 8192), (1, 11, 8192)),
    ],
)
@pytest.mark.push
def test_multiply(forge_property_recorder, shapes):
    shape1, shape2 = shapes

    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x * y

    inputs = [
        torch.rand(shape1),
        torch.rand(shape2),
    ]

    framework_model = Multiply()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 11, 2048),
    ],
)
@pytest.mark.push
def test_reduce_avg(forge_property_recorder, shapes):
    class ReduceAvg(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.mean(x, dim=-1, keepdim=True)

    inputs = [torch.rand(shapes)]

    framework_model = ReduceAvg()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99)),
        forge_property_handler=forge_property_recorder,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 11, 8192),
    ],
)
@pytest.mark.push
def test_sigmoid(forge_property_recorder, shapes):
    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    inputs = [torch.rand(shapes)]

    framework_model = Sigmoid()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 11, 1),),
    ],
)
@pytest.mark.push
def test_reciprocal(forge_property_recorder, shapes):
    class Reciprocal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reciprocal(x)

    inputs = [torch.rand(shapes[0]) + 0.1]

    framework_model = Reciprocal()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "source_and_target_shape",
    [
        ((1, 11, 2048), (11, 2048)),
        ((1, 11, 2048), (1, 11, 32, 64)),
        ((1, 11, 2048), (1, 11, 2048)),
        ((1, 32, 11, 64), (32, 11, 64)),
        ((1, 32, 11, 64), (1, 32, 11, 64)),
        ((11, 512), (1, 11, 8, 64)),
        ((1, 8, 4, 11, 64), (32, 11, 64)),
        ((1, 8, 4, 11, 64), (1, 32, 11, 64)),
        ((32, 11, 11), (1, 32, 11, 11)),
        ((32, 11, 11), (32, 11, 11)),
        ((1, 32, 64, 11), (32, 64, 11)),
        ((1, 11, 32, 64), (11, 2048)),
        ((11, 8192), (1, 11, 8192)),
    ],
)
@pytest.mark.push
def test_reshape(forge_property_recorder, source_and_target_shape):
    source_shape, target_shape = source_and_target_shape

    class Reshape(nn.Module):
        def __init__(self, target_shape):
            super().__init__()
            self.target_shape = target_shape

        def forward(self, a):
            return torch.reshape(a, self.target_shape)

    inputs = [torch.rand(source_shape, dtype=torch.bfloat16)]

    framework_model = Reshape(target_shape)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 32, 11, 11), -1),
    ],
)
@pytest.mark.push
def test_softmax(forge_property_recorder, shapes):
    shape, dim = shapes

    class Softmax(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.softmax(x, dim=dim)

    inputs = [torch.rand(shape)]

    framework_model = Softmax(dim)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 11, 1),
    ],
)
@pytest.mark.push
def test_sqrt(forge_property_recorder, shapes):
    class Sqrt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)

    inputs = [torch.rand(shapes)]
    framework_model = Sqrt()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((32), 0),
        ((1, 11, 64), 1),
    ],
)
@pytest.mark.push
def test_unsqueeze(forge_property_recorder, input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    class Unsqueeze(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, a):
            return torch.unsqueeze(a, self.dim)

    inputs = [torch.rand(input_shape, dtype=torch.bfloat16)]

    framework_model = Unsqueeze(dim)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "params",
    [
        ((2048, 2048), (-2, -1)),
        ((1, 11, 32, 64), (-3, -2)),
        ((1, 32, 11), (-2, -1)),
        ((512, 2048), (-2, -1)),
        ((1, 11, 8, 64), (-3, -2)),
        ((32, 11, 64), (-2, -1)),
        ((32, 11, 64), (-3, -2)),
        ((32, 64, 11), (-2, -1)),
        ((8192, 2048), (-2, -1)),
        ((2048, 8192), (-2, -1)),
        pytest.param(
            ((128256, 2048), (-2, -1)),
            marks=pytest.mark.skip(
                reason="Due to large host memory consumption, it causes flakiness in CI. Issue #1502"
            ),
        ),
    ],
)
@pytest.mark.push
def test_transpose(forge_property_recorder, params):
    shapes, dims = params

    class Transpose(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, a):
            return torch.transpose(a, *self.dims)

    inputs = [torch.rand(shapes)]
    framework_model = Transpose(dims)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
