# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import jax
import flax.linen as nn
import jax.random as random
import jax.numpy as jnp

import forge
from forge.verify.verify import verify
from test.mlir.utils import prepare_jax_test


@pytest.mark.push
def test_add(forge_property_recorder):
    class Add(nn.Module):
        @nn.compact
        def __call__(self, x, y):
            return x + y

    framework_model = Add()
    framework_model, inputs = prepare_jax_test(
        framework_model, [(random.uniform, (2, 32, 32), jnp.float32), (random.uniform, (2, 32, 32), jnp.float32)]
    )
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_arithmetic(forge_property_recorder):
    class Arithmetic(nn.Module):
        @nn.compact
        def __call__(self, x, y):
            return jax.numpy.sqrt(x) + jax.numpy.exp(y)

    framework_model = Arithmetic()
    framework_model, inputs = prepare_jax_test(
        framework_model, [(random.uniform, (2, 32, 32), jnp.float32), (random.uniform, (2, 32, 32), jnp.float32)]
    )

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_matmul(forge_property_recorder):
    class Matmul(nn.Module):
        @nn.compact
        def __call__(self, x, y):
            return jnp.matmul(x, y)

    framework_model = Matmul()
    framework_model, inputs = prepare_jax_test(
        framework_model, [(random.uniform, (32, 64), jnp.float32), (random.uniform, (64, 32), jnp.float32)]
    )

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_squeeze(forge_property_recorder):
    class Squeeze(nn.Module):
        @nn.compact
        def __call__(self, x, y):
            squeezed_x = jnp.squeeze(x, axis=0)
            squeezed_y = jnp.squeeze(y, axis=0)
            return squeezed_x.T + squeezed_y

    framework_model = Squeeze()
    framework_model, inputs = prepare_jax_test(
        framework_model, [(random.uniform, (1, 32, 32), jnp.float32), (random.uniform, (1, 32, 32), jnp.float32)]
    )

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_flatten(forge_property_recorder):
    class Flatten(nn.Module):
        @nn.compact
        def __call__(self, x):
            return jnp.reshape(x, (x.shape[0], -1))

    framework_model = Flatten()
    framework_model, inputs = prepare_jax_test(framework_model, [(random.uniform, (2, 32, 32), jnp.float32)])

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_linear_layer(forge_property_recorder):
    class Linear(nn.Module):
        @nn.compact
        def __call__(self, x):
            dense = nn.Dense(features=10, use_bias=True)
            return dense(x)

    framework_model = Linear()
    framework_model, inputs = prepare_jax_test(framework_model, [(random.uniform, (1, 784), jnp.float32)])

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
@pytest.mark.xfail(reason="unsupported broadcast operation in ttnn at runtime")
def test_multiple_layers(forge_property_recorder):
    class CNNClassifier(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Convert from NCHW to NHWC format
            #   (N, C, H, W) -> (N, H, W, C)
            x = jnp.transpose(x, (0, 2, 3, 1))

            x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

            x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(features=128)(x)
            x = nn.relu(x)
            x = nn.Dense(features=10)(x)
            return x

    framework_model = CNNClassifier()
    framework_model, inputs = prepare_jax_test(framework_model, [(random.uniform, (1, 3, 32, 32), jnp.float32)])

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
@pytest.mark.skip(reason="Conversion for ttnn::relu is missing and the maximum operation is incorrect")
def test_mnist_linear(forge_property_recorder):
    class MNISTLinear(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=512)(x)
            x = nn.relu(x)
            x = nn.Dense(features=512)(x)
            x = nn.relu(x)
            x = nn.Dense(features=10)(x)
            return x

    framework_model = MNISTLinear()
    framework_model, inputs = prepare_jax_test(framework_model, [(random.uniform, (1, 784), jnp.float32)])

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_batchnorm(forge_property_recorder):
    class BatchNorm(nn.Module):
        @nn.compact
        def __call__(self, x):
            batch_norm = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5, dtype=jnp.float32)
            return batch_norm(x)

    framework_model = BatchNorm()
    framework_model, inputs = prepare_jax_test(framework_model, [(random.uniform, (1, 32, 56, 56), jnp.float32)])

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_convbn(forge_property_recorder):
    class ConvBNLayer(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.BatchNorm(
                use_running_average=True,
                momentum=0.9,
                epsilon=1e-5,
            )(x)
            return x

    framework_model = ConvBNLayer()
    framework_model, inputs = prepare_jax_test(framework_model, [(random.normal, (1, 64, 64, 3), jnp.float32)])

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
