# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import os
import onnx
import tensorflow as tf

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.nightly
def test_lstm_genom_onnx(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX, model="lstm_genom", source=Source.GITHUB, task=Task.SEQUENCE_MODELING
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Run inference on Tenstorrent device
    inputs = [tf.random.uniform(shape=[1, 10, 4])]

    # Load ONNX model
    load_path = "third_party/confidential_customer_models/model_2/onnx/lstm_genom/lstm-genom-model.onnx"
    model_name = f"lstm_genom_onnx"
    onnx_model = onnx.load(load_path)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
