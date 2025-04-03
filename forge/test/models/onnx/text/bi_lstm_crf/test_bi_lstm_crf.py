# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.verify.verify import verify

from test.models.onnx.text.bi_lstm_crf.utils.model import get_model
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.xfail()
def test_birnn_crf_pypi(forge_property_recorder, tmp_path):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="BiRnnCrf_PyPI",
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    test_sentence = ["apple", "corporation", "is", "in", "georgia"]

    # Load model and input tensor
    model, test_input = get_model(test_sentence)
    model.eval()

    onnx_path = f"{tmp_path}/bilstm_crf.onnx"
    torch.onnx.export(model, test_input, onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        model, sample_inputs=(test_input,), module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(test_input, model, compiled_model, forge_property_handler=forge_property_recorder)

    
