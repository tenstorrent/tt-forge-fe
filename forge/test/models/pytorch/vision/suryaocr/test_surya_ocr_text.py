# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import shutil

import pytest
import surya.common.surya.processor as _surya_processor_module
from third_party.tt_forge_models.suryaocr.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import _process_image_input

_surya_processor_module.SuryaOCRProcessor._process_image_input = _process_image_input


@pytest.mark.nightly
@pytest.mark.xfail
def test_surya_ocr():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SURYAOCR,
        variant="ocr_text",
        task=Task.OPTICAL_CHARACTER_RECOGNITION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs via loader
    loader = ModelLoader(variant=ModelVariant.OCR_TEXT)
    inputs = loader.load_inputs()
    framework_model = loader.load_model()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post process outputs
    output_dir = "test/models/pytorch/vision/suryaocr/surya_text"
    try:
        loader.post_process(co_out, output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
