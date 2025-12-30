# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from gliner import GLiNER

import forge
from forge.verify.verify import verify
import onnx

from test.models.onnx.text.gliner.model_utils.model_utils import export_onnx, prepare_inputs
from forge.forge_property_utils import Framework, ModelArch, Source, Task, record_model_properties

variants = [pytest.param("urchade/gliner_multi-v2.1", marks=pytest.mark.pr_models_regression)]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_gliner_onnx(variant, forge_tmp_path):

    # Build Module Name
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GLINER,
        variant=variant,
        task=Task.NLP_TOKEN_CLASSIFICATION,
        source=Source.GITHUB,
    )

    pytest.xfail(reason="Segmentation Fault")

    # Load model
    framework_model = GLiNER.from_pretrained(variant)
    framework_model.eval()

    # prepare input
    text = """
    Cristiano Ronaldo dos Santos Aveiro was born 5 February 1985 is a Portuguese professional footballer.
    """
    labels = ["person", "award", "date", "competitions", "teams"]
    inputs = prepare_inputs(framework_model, [text], labels)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/gliner.onnx"
    export_onnx(tuple(inputs), framework_model, onnx_path)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
