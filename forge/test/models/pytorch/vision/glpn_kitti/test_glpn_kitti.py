# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.vision.glpn_kitti.model_utils.utils import (
    load_input,
    load_model,
)


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
def test_glpn_kitti(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="glpn_kitti",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.DEPTH_ESTIMATION,
    )

    # Load model and input
    framework_model = load_model(variant)
    inputs = load_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
