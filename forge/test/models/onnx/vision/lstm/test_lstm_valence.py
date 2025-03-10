# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import os
import onnx
import tensorflow as tf
import forge
from forge.verify.backend import verify_module
from forge import DepricatedVerifyConfig
from forge._C.backend_api import BackendType, BackendDevice
from forge.verify.config import TestKind


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Not supported")
@pytest.mark.nightly
def test_lstm_valence_onnx(test_device):
    # Load model checkpoint from HuggingFace
    load_path = "third_party/confidential_customer_models/model_2/onnx/lstm_valence/lstm-valence-model.onnx"
    model = onnx.load(load_path)

    # Set Forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    # Required to patch data-mismatch. Here is followup issue
    # to check this out in more details:
    # tenstorrent/forge#1828
    os.environ["FORGE_DECOMPOSE_SIGMOID"] = "1"

    # Run inference on Tenstorrent device
    inputs = tf.random.uniform(shape=[1, 1, 282])
    verify_module(
        forge.OnnxModule("onnx_lstm", model),
        input_shapes=(inputs.shape,),
        inputs=[(inputs,)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.98,
        ),
    )
