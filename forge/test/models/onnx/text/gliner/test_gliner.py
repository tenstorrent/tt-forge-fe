# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from gliner import GLiNER

import forge
from forge.verify.verify import verify
import onnx

from test.models.onnx.text.gliner.utils.model_utils import export_onnx, prepare_inputs
from test.models.utils import Framework, Source, Task, build_module_name

variants = ["urchade/gliner_multi-v2.1"]


@pytest.mark.nightly
# @pytest.mark.skip(reason="Segmentation Fault")
@pytest.mark.parametrize("variant", variants)
def test_gliner_onnx(forge_property_recorder, variant, tmp_path):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Gliner",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    # Load model
    framework_model = GLiNER.from_pretrained(variant)
    framework_model.eval()

    # prepare input
    text = """
    Cristiano Ronaldo dos Santos Aveiro was born 5 February 1985) is a Portuguese professional footballer.
    """
    labels = ["person", "award", "date", "competitions", "teams"]
    inputs, inputs_forge = prepare_inputs(framework_model, [text], labels)

    # Export model to ONNX
    onnx_path = f"{tmp_path}/gliner.onnx"
    export_onnx(inputs, framework_model, onnx_path)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs_forge, forge_property_handler=forge_property_recorder
    )
