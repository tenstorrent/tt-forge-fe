# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

import forge
from forge.config import CompilerConfig
from forge.tensor import to_pt_tensors
from forge.verify.verify import verify
from forge._C import DataFormat


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w",
    (
        (1, 64, 16, 115, 115, 4, 4, 1, 1),
        (1, 64, 64, 56, 56, 3, 3, 1, 1),
        (1, 128, 128, 56, 56, 3, 3, 2, 2),
        (1, 128, 128, 28, 28, 3, 3, 1, 1),
        (1, 256, 256, 28, 28, 3, 3, 2, 2),
        (1, 256, 256, 14, 14, 3, 3, 1, 1),
        (1, 64, 64, 8, 8, 3, 3, 1, 1),
        (1, 64, 64, 16, 16, 3, 3, 1, 1),
        (1, 256, 256, 7, 7, 3, 3, 1, 1),
        (1, 256, 64, 56, 56, 1, 1, 2, 2),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        pytest.param(
            tf.bfloat16, marks=pytest.mark.xfail(reason="dtypes are not properly lowered from tensorflow #281")
        ),
        tf.float32,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [
        tf.bfloat16,
        tf.float32,
    ],
)
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.push
def test_conv2d(
    forge_property_recorder,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    activations_dtype,
    weights_dtype,
    has_bias,
):
    tf.random.set_seed(0)
    if (
        activations_dtype == tf.float32
        and weights_dtype == tf.float32
        and input_height == input_width == 28
        and input_channels == output_channels == 256
    ):
        pytest.skip("Circular buffer grows beyond maximum L1 size.")

    padding = "same" if stride_h == stride_w == 1 and filter_height % 2 == 1 else "valid"

    class Conv2d(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv2d = tf.keras.layers.Conv2D(
                output_channels,
                (filter_height, filter_width),
                strides=[stride_h, stride_w],
                padding=padding,
                data_format="channels_last",
                dtype=weights_dtype,
                use_bias=has_bias,
            )

        def call(self, x):
            return self.conv2d(x)

    inputs = [tf.random.uniform((batch_size, input_height, input_width, input_channels), dtype=activations_dtype)]

    framework_model = Conv2d()

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_dual_conv2d(forge_property_recorder):

    tf.random.set_seed(0)

    class DualConv2d(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv2d1 = tf.keras.layers.Conv2D(
                64, (3, 3), strides=[1, 1], padding="same", data_format="channels_last"
            )
            self.conv2d2 = tf.keras.layers.Conv2D(
                256, (2, 2), strides=[2, 2], padding="valid", data_format="channels_last"
            )

        def call(self, x):
            return self.conv2d2(self.conv2d1(x))

    inputs = [tf.random.uniform((1, 128, 128, 3))]

    framework_model = DualConv2d()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "act_shape",  ## NHWC
    [
        (1, 32, 32, 32),
        (1, 32, 32, 64),
        (1, 32, 32, 128),
        (1, 32, 64, 32),
        (1, 32, 64, 64),
        (1, 32, 64, 128),
        (1, 32, 128, 32),
        (1, 32, 128, 64),
        (1, 32, 128, 128),
        (1, 64, 32, 32),
        (1, 64, 32, 64),
        (1, 64, 32, 128),
        (1, 64, 64, 32),
        (1, 64, 64, 64),
        (1, 64, 64, 128),
        (1, 64, 128, 32),
        (1, 64, 128, 64),
        (1, 64, 128, 128),
        (1, 128, 32, 32),
        (1, 128, 32, 64),
        (1, 128, 32, 128),
        (1, 128, 64, 32),
        (1, 128, 64, 64),
        (1, 128, 64, 128),
        (1, 128, 128, 32),
        (1, 128, 128, 64),
        (1, 128, 128, 128),
    ],
)
@pytest.mark.push
def test_maxpool2d(
    forge_property_recorder,
    act_shape,
):
    # NOTE: Only shapes that are tile-dim aligned before and after
    # the maxpool operation work through the forge-mlir flow. This,
    # limits the variablity that is supported for padding, pool size,
    # strides, and dilation.
    tf.random.set_seed(0)

    class MaxPool(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="valid", dtype=tf.bfloat16)
            self.conv = tf.keras.layers.Conv2D(act_shape[-1], (3, 3), padding="same", dtype=tf.bfloat16)

        def call(self, x):
            x = self.pool(x)
            x = self.conv(x)
            return x

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)
    inputs = [tf.random.uniform(act_shape, dtype=tf.bfloat16)]

    framework_model = MaxPool()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_batch_norm(forge_property_recorder):
    tf.random.set_seed(0)

    class TFBatchNorm(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.batch_norm = tf.keras.layers.BatchNormalization()

        def call(self, x):
            return self.batch_norm(x)

    inputs = [tf.random.uniform((1, 56, 56, 32), dtype=tf.float32)]

    framework_model = TFBatchNorm()

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
