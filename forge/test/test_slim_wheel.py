# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import flax.linen as nn
import onnx
import torch
import tensorflow as tf
import pytest

import forge
from forge.verify.verify import verify


@pytest.mark.slim_wheel
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((4, 4), torch.float32),
        ((6, 7), torch.float32),
        ((2, 3, 4), torch.float32),
    ],
)
def test_eltwise_add_pt(shape, dtype):
    """Test element-wise addition using forge compile and verify."""

    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    input1 = torch.randn(shape, dtype=dtype)
    input2 = torch.randn(shape, dtype=dtype)
    inputs = [input1, input2]

    model = AddModel()
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.slim_wheel
def test_eltwise_add_onnx(forge_tmp_path):
    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    # Load model
    torch_model = AddModel()
    torch_model.eval()

    # Load input
    input1 = torch.randn(1, 3, 224, 224)
    input2 = torch.randn(1, 3, 224, 224)
    inputs = [input1, input2]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/eltwise_add.onnx"
    torch.onnx.export(torch_model, (input1, input2), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule("eltwise_add", onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.slim_wheel
def test_eltwise_add_tf():
    class AddModel(tf.keras.Model):
        def call(self, x, y):
            return x + y

    # Load framework model
    framework_model = AddModel()

    # Load sample inputs
    input1 = tf.random.normal([1, 3, 224, 224])
    input2 = tf.random.normal([1, 3, 224, 224])
    inputs = [input1, input2]

    # Compile model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify data on sample input
    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.slim_wheel
def test_eltwise_add_jax():
    class AddModel(nn.Module):
        @nn.compact
        def __call__(self, x, y):
            return x + y

    # Create model and inputs
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num=3)  # One for init, two for inputs

    framework_model = AddModel()
    input1 = jax.random.normal(keys[1], (1, 3, 224, 224))
    input2 = jax.random.normal(keys[2], (1, 3, 224, 224))
    inputs = [input1, input2]

    # Initialize and bind variables to the model
    variables = framework_model.init(keys[0], input1, input2)
    bound_model = framework_model.bind(variables)

    # Compile model
    compiled_model = forge.compile(bound_model, inputs)

    # Verify data on sample input
    verify(inputs, bound_model, compiled_model)
