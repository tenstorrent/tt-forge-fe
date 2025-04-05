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


@pytest.mark.push
def test_add(forge_property_recorder):
    class Add(nn.Module):
        @nn.compact
        def __call__(self, x, y):
            return x + y

    key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    inputs = [random.uniform(key1, shape=(2, 32, 32)), random.uniform(key2, shape=(2, 32, 32))]

    framework_model = Add()
    variables = framework_model.init(key, inputs[0], inputs[1])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    inputs = [random.uniform(key1, shape=(2, 32, 32)), random.uniform(key2, shape=(2, 32, 32))]

    framework_model = Arithmetic()
    variables = framework_model.init(key, inputs[0], inputs[1])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    inputs = [random.uniform(key1, shape=(32, 64)), random.uniform(key2, shape=(64, 32))]

    framework_model = Matmul()
    variables = framework_model.init(key, inputs[0], inputs[1])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    inputs = [random.uniform(key1, shape=(1, 32, 32)), random.uniform(key2, shape=(1, 32, 32))]

    framework_model = Squeeze()
    variables = framework_model.init(key, inputs[0], inputs[1])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    inputs = [random.uniform(key, shape=(2, 32, 32))]

    framework_model = Flatten()
    variables = framework_model.init(key, inputs[0])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    inputs = [random.uniform(key, shape=(1, 784))]

    framework_model = Linear()
    variables = framework_model.init(key, inputs[0])
    framework_model = framework_model.bind(variables)

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
@pytest.mark.skip(reason="unsupported broadcast operation in ttnn at runtime")
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

    key = random.PRNGKey(0)
    inputs = [random.uniform(key, shape=(1, 3, 32, 32))]  # NCHW format

    framework_model = CNNClassifier()
    variables = framework_model.init(key, inputs[0])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    inputs = [random.uniform(key, shape=(1, 784))]

    framework_model = MNISTLinear()
    variables = framework_model.init(key, inputs[0])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    inputs = [random.uniform(key, shape=(1, 32, 56, 56))]

    framework_model = BatchNorm()
    variables = framework_model.init(key, inputs[0])
    framework_model = framework_model.bind(variables)

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

    key = random.PRNGKey(0)
    inputs = [random.normal(key, shape=(1, 64, 64, 3))]

    framework_model = ConvBNLayer()
    variables = framework_model.init(key, inputs[0])
    framework_model = framework_model.bind(variables)

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
