# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.glpn_kitti.utils.utils import load_input, load_model
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "vinvino02/glpn-kitti",
            marks=[pytest.mark.skip(reason="Only tilized tensors are supported for device typecast")],
        ),
    ],
)
def test_glpn_kitti(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="glpn_kitti",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.DEPTH_ESTIMATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load model and input
    framework_model = load_model(variant)
    inputs = load_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
