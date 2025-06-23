# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from third_party.tt_forge_models.centernet import ModelLoader


@pytest.mark.nightly
@pytest.mark.push
def test_centernet():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.CENTERNET,
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
    )

    # Load model and input
    framework_model = ModelLoader.load_model()
    input_sample = ModelLoader.load_inputs()

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.enable_optimization_passes = True

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_sample],
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify([input_sample], framework_model, compiled_model)
