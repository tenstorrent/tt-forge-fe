# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from third_party.tt_forge_models.vgg19_unet import ModelLoader  # isort:skip


@pytest.mark.nightly
@pytest.mark.xfail
def test_vgg19_unet(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="VGG19 UNet",
        variant="default",
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.GITHUB,
    )

    # Load model and input
    framework_model = ModelLoader.load_model()
    input_sample = ModelLoader.load_inputs()

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_sample],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify([input_sample], framework_model, compiled_model, forge_property_handler=forge_property_recorder)
