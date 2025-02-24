# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.dla.utils.utils import load_dla_model, post_processing
from test.models.utils import Framework, Source, Task, build_module_name

variants = [
    pytest.param("dla34", marks=[pytest.mark.push]),
    "dla46_c",
    "dla46x_c",
    "dla60",
    "dla60x",
    "dla60x_c",
    "dla102",
    "dla102x",
    "dla102x2",
    "dla169",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_pytorch(record_forge_property, variant):
    if variant != "dla34":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="dla", variant=variant, task=Task.VISUAL_BACKBONE, source=Source.TORCHVISION
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load the model and prepare input data
    framework_model, inputs = load_dla_model(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    # Inference
    output = compiled_model(*inputs)

    # post processing
    post_processing(output)
