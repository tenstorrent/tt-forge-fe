# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import shutil

import pytest
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


@pytest.mark.nightly
@pytest.mark.xfail
def test_surya_ocr():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SURYAOCR,
        variant="ocr_detection",
        task=Task.OPTICAL_CHARACTER_RECOGNITION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and inputs via loader
    loader = ModelLoader(variant=ModelVariant.OCR_DETECTION)
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
    output_dir = "test/models/pytorch/vision/suryaocr/surya_detect"
    try:
        loader.post_process(co_out, output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
