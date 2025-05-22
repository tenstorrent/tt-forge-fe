# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.vision.monodepth2.model_utils.utils import (
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
def test_monodepth2(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="monodepth2",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.DEPTH_PREDICTION,
    )

    # prepare model and input
    download_model(variant)
    framework_model, height, width = load_model(variant)
    input_tensor = load_input(height, width)

    inputs = [input_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
