# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import forge
from test.models.pytorch.vision.monodepth2.utils.utils import download_model, load_model, load_input
from test.models.utils import build_module_name

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
    # prepare model and input
    download_model(variant)
    model, height, width = load_model(variant)
    input_tensor = load_input(height, width)

    # Forge inference
    module_name = build_module_name(framework="pt", model="monodepth2", variant=variant)
    compiled_model = forge.compile(model, sample_inputs=[input_tensor], module_name=module_name)
