# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.ghostnet.utils.utils import (
    load_ghostnet_model,
    post_processing,
)
from test.models.utils import Framework, Source, Task, build_module_name

params = [
    pytest.param("ghostnet_100", marks=[pytest.mark.push]),
    pytest.param("ghostnet_100.in1k", marks=[pytest.mark.push]),
    pytest.param("ghostnetv2_100.in1k"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", params)
def test_ghostnet_timm(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="ghostnet",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load the model and prepare input data
    framework_model, inputs = load_ghostnet_model(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)
