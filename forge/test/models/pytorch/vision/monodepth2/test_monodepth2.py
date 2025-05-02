# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.monodepth2.utils.utils import (
    download_model,
    load_input,
    load_model,
)

variants = [
    "mono_640x192",
    "stereo_640x192",
    "mono+stereo_640x192",
    "mono_no_pt_640x192",
    "stereo_no_pt_640x192",
    "mono+stereo_no_pt_640x192",
    "mono_1024x320",
    "stereo_1024x320",
    "mono+stereo_1024x320",
]


@pytest.mark.parametrize("variant", variants)
def test_monodepth2(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="monodepth2",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.DEPTH_PREDICTION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # prepare model and input
    download_model(variant)
    framework_model, height, width = load_model(variant)
    input_tensor = load_input(height, width)

    inputs = [input_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
