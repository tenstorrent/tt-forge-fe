# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from third_party.tt_forge_models.suryaocr.pytorch import ModelLoader

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
        variant="default",
        task=Task.OPTICAL_CHARACTER_RECOGNITION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and inputs via loader
    loader = ModelLoader()
    framework_model = loader.load_model()
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
