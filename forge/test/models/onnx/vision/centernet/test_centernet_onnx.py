# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from third_party.tt_forge_models.centernet.onnx import ModelLoader
import onnx
import forge
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.xfail
def test_centernet_onnx():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.CENTERNET,
        task=Task.CV_OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Load model and input
    loader = ModelLoader()
    onnx_model = loader.load_model()
    inputs = loader.load_inputs()

    # Load framework model
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, [inputs], module_name=module_name)

    # Model Verification and Inference
    verify(
        [inputs],
        framework_model,
        compiled_model,
    )
