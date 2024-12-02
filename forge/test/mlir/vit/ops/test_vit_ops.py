# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden_pcc


@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 768, 14, 14), (768, 1, 1)],
        [(1, 197, 768), (1, 197, 768)],
        [(1, 197, 768), (768,)],
        [(1, 197, 3072), (3072,)],
        [(1, 768), (768,)],
    ],
)
@pytest.mark.push
def test_add(shapes):
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [torch.rand(shapes[0]), torch.rand(shapes[1])]  # when we use dtype=torch.bfloat16, pcc fails

    framework_model = Add()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.push
@pytest.mark.parametrize(
    "shapes",
    [
        (1, 1, 768),
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        (-3),
    ],
)
@pytest.mark.parametrize(
    "new_shape",
    [
        (1),
    ],
)
def test_broadcast(shapes, dim, new_shape):
    class Broadcast(nn.Module):
        def __init__(self, dim, new_shape):
            super().__init__()
            self.dim = dim
            self.new_shape = new_shape

        def forward(self, x):
            # Get the size of x
            x_size = list(x.size())

            # Calculate the new shape for the x
            x_size[self.dim] = self.new_shape

            # Expand the tensor to the new shape
            broadcasted_tensor = x.expand(x_size)

            return broadcasted_tensor

    inputs = [torch.rand(shapes)]

    # Framework Model
    framework_model = Broadcast(dim, new_shape)
    fw_out = framework_model(*inputs)

    # Compile Model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Move compiled outputs to CPU for comparison
    co_out = [co.to("cpu") for co in co_out]

    # Validate the output shapes and values
    assert fw_out.shape == co_out[0].shape, f"Expected shape {fw_out.shape}, but got {co_out[0].shape}"
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


# E       RuntimeError: TT_FATAL @ /proj_sw/user_dev/vkovinic/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp:47: !in_ref.get_shape().has_tile_padding(this->dim)
# E       info:
# E       Tile padding along concatenated dim (2) not supported for concat yet (tensor: 0).
@pytest.mark.parametrize(
    "inputs_and_dim",
    [
        ((1, 1, 768), (1, 196, 768), -2),
    ],
)
@pytest.mark.push
def test_concat(inputs_and_dim):
    in_shape1, in_shape2, dim = inputs_and_dim

    class Concat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.cat((a, b), dim)

    inputs = [torch.rand(in_shape1), torch.rand(in_shape2)]

    framework_model = Concat()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


# error: 'ttnn.conv2d' op Bias must only have data on the final dimenstion
# #637 Issue
@pytest.mark.parametrize("shape", [(1, 3, 224, 224)])
@pytest.mark.parametrize(
    "conv_params",
    [
        {
            "out_channels": 768,
            "kernel_size": (16, 16),
            "stride": (16, 16),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        }
    ],
)
@pytest.mark.push
def test_conv2d(shape, conv_params):
    class Conv2d(nn.Module):
        def __init__(self, conv_params):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=shape[1],
                out_channels=conv_params["out_channels"],
                kernel_size=conv_params["kernel_size"],
                stride=conv_params["stride"],
                padding=conv_params["padding"],
                dilation=conv_params["dilation"],
                groups=conv_params["groups"],
            )

        def forward(self, x):
            return self.conv(x)

    # Instantiate the model with the parameters
    framework_model = Conv2d(conv_params)

    # Prepare the input tensor
    inputs = [torch.rand(shape)]

    # Get framework output
    fw_out = framework_model(*inputs)

    # Compile the model with Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Move compiled output to CPU for comparison
    co_out = [co.to("cpu") for co in co_out]
    fw_out = fw_out if isinstance(fw_out, list) else [fw_out]

    # Ensure the framework and compiled outputs match
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


# 2024-11-25 14:45:04.620 | ERROR    | Always          - Found Unsupported operations while lowering from TTForge to TTIR in forward graph
# Unsupported Ops at: RUN_MLIR_COMPILER stage
# gelu
#                  Input_shape: [{1, 197, 3072}, ]
#                                          Attributes: gelu(none,)
# #745 Issue
@pytest.mark.parametrize("shape", [(1, 197, 3072)])
@pytest.mark.parametrize(
    "gelu_params",
    [{"approximate": "none"}],
)
@pytest.mark.push
def test_gelu(shape, gelu_params):
    class GELU(nn.Module):
        def __init__(self, gelu_params):
            super().__init__()
            # GELU can be used from PyTorch as nn.GELU
            self.gelu = nn.GELU(approximate=gelu_params["approximate"])

        def forward(self, x):
            return self.gelu(x)

    # Instantiate the model with the parameters
    framework_model = GELU(gelu_params)

    # Prepare the input tensor
    inputs = [torch.rand(shape)]

    # Get framework output
    fw_out = framework_model(*inputs)

    # Compile the model with Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Move compiled output to CPU for comparison
    co_out = [co.to("cpu") for co in co_out]
    fw_out = fw_out if isinstance(fw_out, list) else [fw_out]

    # Ensure the framework and compiled outputs match
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize("shape", [(1, 197, 768)])
@pytest.mark.parametrize(
    "index_params",
    [
        {
            "dim": -2,  # Slice the second-to-last dimension (i.e., index dimension -2)
            "start": 0,  # Start index for slicing
            "stop": 1,  # End index (exclusive) for slicing
            "stride": 1,  # Stride for slicing
        }
    ],
)
@pytest.mark.push
def test_index(shape, index_params):
    class IndexModule(nn.Module):
        def __init__(self, index_params):
            super().__init__()
            self.dim = index_params["dim"]
            self.start = index_params["start"]
            self.stop = index_params["stop"]
            self.stride = index_params["stride"]

        def forward(self, x):
            return x.narrow(self.dim, self.start, self.stop - self.start)[:: self.stride]

    # Instantiate the model with the parameters
    framework_model = IndexModule(index_params)

    # Prepare the input tensor
    inputs = [torch.rand(shape)]

    # Get framework output
    fw_out = framework_model(*inputs)

    # Compile the model with Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Move compiled output to CPU for comparison
    co_out = [co.to("cpu") for co in co_out]
    fw_out = fw_out if isinstance(fw_out, list) else [fw_out]

    # Ensure the framework and compiled outputs match
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize("shape", [(1, 197, 768)])
@pytest.mark.parametrize(
    "layernorm_params",
    [
        {
            "weights": torch.rand(768),  # The shape of the weights tensor should match the last dimension of the input
            "bias": torch.rand(768),  # The shape of the bias tensor should match the last dimension of the input
            "dim": -1,  # Normalize over the last dimension (usually feature dimension)
            "epsilon": 1e-5,  # Small epsilon for numerical stability
        }
    ],
)
@pytest.mark.push
def test_layernorm(shape, layernorm_params):
    class LayernormModule(nn.Module):
        def __init__(self, layernorm_params):
            super().__init__()
            self.weights = layernorm_params["weights"]
            self.bias = layernorm_params["bias"]
            self.dim = layernorm_params["dim"]
            self.epsilon = layernorm_params["epsilon"]

        def forward(self, x):
            return nn.functional.layer_norm(x, (x.size(self.dim),), self.weights, self.bias, self.epsilon)

    # Instantiate the model with the parameters
    framework_model = LayernormModule(layernorm_params)

    # Prepare the input tensor
    inputs = [torch.rand(shape)]

    # Get framework output
    fw_out = framework_model(*inputs)

    # Compile the model with Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Move compiled output to CPU for comparison
    co_out = [co.to("cpu") for co in co_out]
    fw_out = fw_out if isinstance(fw_out, list) else [fw_out]

    # Ensure the framework and compiled outputs match
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "shapes",
    [
        ((197, 768), (768, 768)),
        ((12, 197, 64), (12, 64, 197)),
        ((12, 197, 197), (12, 197, 64)),
        ((1, 197, 768), (768, 3072)),
        ((1, 197, 3072), (3072, 768)),
        ((1, 768), (768, 768)),
    ],
)
@pytest.mark.push
def test_matmul(shapes):
    shape1, shape2 = shapes

    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    inputs = [
        torch.rand(shape1),
        torch.rand(shape2),
    ]

    framework_model = Matmul()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.95)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 12, 197, 197), (1)),
    ],
)
@pytest.mark.push
def test_multiply(shapes):

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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


@pytest.mark.parametrize(
    "source_and_target_shape",
    [
        ((1, 768, 14, 14), (1, 768, 196, 1)),
        ((1, 197, 768), (197, 768)),
        ((1, 197, 768), (1, 197, 768)),
        ((1, 197, 768), (1, 197, 12, 64)),
        ((1, 12, 197, 64), (12, 197, 64)),
        ((1, 12, 197, 64), (1, 12, 197, 64)),
        ((12, 197, 197), (1, 12, 197, 197)),
        ((12, 197, 197), (12, 197, 197)),
        ((1, 12, 64, 197), (12, 64, 197)),
        ((1, 197, 12, 64), (197, 768)),
        ((1, 1, 768), (1, 768)),
    ],
)
@pytest.mark.push
def test_reshape(source_and_target_shape):
    source_shape, target_shape = source_and_target_shape

    if len(source_shape) > 4 or len(target_shape) > 4:
        pytest.xfail("Only 2D, 3D, and 4D tensors are supported")

    if (
        source_and_target_shape == ((32, 11, 11), (1, 32, 11, 11))
        or source_and_target_shape == ((32, 11, 11), (32, 11, 11))
        or source_and_target_shape == ((1, 32, 64, 11), (32, 64, 11))
    ):
        pytest.xfail("pcc < 0.99")

    class Reshape(nn.Module):
        def __init__(self, target_shape):
            super().__init__()
            self.target_shape = target_shape

        def forward(self, a):
            return torch.reshape(a, self.target_shape)

    inputs = [torch.rand(source_shape, dtype=torch.bfloat16)]
    framework_model = Reshape(target_shape)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)
    # some of them are failing with pcc < 0.99


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 12, 197, 197), -1),
    ],
)
@pytest.mark.push
def test_softmax(shapes):
    shape, dim = shapes

    class Softmax(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.softmax(x, dim=dim)

    inputs = [torch.rand(shape)]
    framework_model = Softmax(dim)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((1, 768, 196, 1), -1),
    ],
)
@pytest.mark.push
def test_squeeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    class Squeeze(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, a):
            return torch.squeeze(a, self.dim)

    inputs = [torch.rand(input_shape)]  # pcc fails if we use dtype=torch.bfloat16

    framework_model = Squeeze(dim)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


#     def run_mlir_compiler(context: CompileContext) -> CompileDepth:
#         assert context.forge_module is not None

# >       context.compiled_binary = forge._C.run_mlir_compiler(context.forge_module)
# E       RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 768),
    ],
)
@pytest.mark.push
def test_tanh(input_shape):
    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.tanh(a)

    inputs = [torch.rand(input_shape)]

    framework_model = Tanh()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "params",
    [
        ((1, 768, 196), (-2, -1)),
        ((768, 768), (-2, -1)),
        ((1, 197, 12, 64), (-3, -2)),
        ((12, 197, 64), (-2, -1)),
        ((12, 197, 64), (-3, -2)),
        ((12, 64, 197), (-2, -1)),
        ((3072, 768), (-2, -1)),
        ((768, 3072), (-2, -1)),
    ],
)
@pytest.mark.push
def test_transpose(params):
    shapes, dims = params

    class Transpose(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, a):
            return torch.transpose(a, *self.dims)

    inputs = [torch.rand(shapes)]
    framework_model = Transpose(dims)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


# context = CompileContext(modules=[Module Unsqueeze], graph_name='Unsqueeze', compiler_cfg=CompilerConfig(enable_training=False, ...unt=0, target_cycles_offset=0, forge_module=<forge._C.ForgeGraphModule object at 0x7f5c2bd22f70>, compiled_binary=None)

#     def run_mlir_compiler(context: CompileContext) -> CompileDepth:
#         assert context.forge_module is not None

# >       context.compiled_binary = forge._C.run_mlir_compiler(context.forge_module)
# E       RuntimeError: Failed to run MLIR compiler pass pipeline.

# forge/forge/compile.py:935: RuntimeError

#  error: 'ttnn.reshape' op Shape attribute size must match output tensor rank
@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((768), 1),
        ((768, 1), 1),
    ],
)
@pytest.mark.push
def test_unsqueeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    class Unsqueeze(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, a):
            return torch.unsqueeze(a, self.dim)

    inputs = [torch.rand(input_shape)]  # pcc fails if we use dtype=torch.bfloat16

    framework_model = Unsqueeze(dim)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)