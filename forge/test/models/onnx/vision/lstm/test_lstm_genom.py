# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import os
import onnx
import tensorflow as tf

# TODO: These are old forge, we should update them to the currently version.
# import forge
# from forge.verify.backend import verify_module
# from forge import DepricatedVerifyConfig
# from forge._C.backend_api import BackendType, BackendDevice
# from forge.verify.config import TestKind

from test.utils import download_model


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.nightly
def test_lstm_genom_onnx(test_device):
    load_path = "third_party/confidential_customer_models/model_2/onnx/lstm_genom/lstm-genom-model.onnx"
    model = onnx.load(load_path)

    # Run inference on Tenstorrent device
    inputs = tf.random.uniform(shape=[1, 10, 4])
    verify_module(
        forge.OnnxModule("onnx_lstm", model),
        input_shapes=(inputs.shape,),
        inputs=[(inputs,)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.95,
        ),
    )
